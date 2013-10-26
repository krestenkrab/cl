
-module(nnagent).
-behavior(gen_server).

-export([init/1, handle_call/3]).

-export([learn/3]).

-include("../include/cl.hrl").

-define(GAMMA, 0.1).

-record(nninput, {
          ibuf    :: undefined | cl:cl_mem(),
          icount  :: undefined | non_neg_integer()
         }).

-record(nnlayer, {
          ibuf    :: undefined | cl:cl_mem(),
          icount  :: undefined | non_neg_integer(),

          obuf    :: undefined | cl:cl_mem(),
          ocount  :: undefined | non_neg_integer(),

          wbuf    :: undefined | cl:cl_mem(),
          wcount  :: undefined | non_neg_integer(),

          dbuf    :: undefined | cl:cl_mem(),

          kernel    :: cl:cl_kernel(),
          backprop  :: cl:cl_kernel(),
          adjust    :: cl:cl_kernel(),
          max_local :: non_neg_integer()
         }).

-record(state, {
          context :: cl:cl_context(),
          input   :: #nninput{},
          layers  :: [ #nnlayer{} ],
          e       :: #cl{},
          queue   :: cl:cl_queue()
         }).

learn(PID, Inputs, Targets) ->
    {ok, Outputs} = gen_server:call(PID, {execute, Inputs}),
    ok = gen_server:call(PID, {correct, Outputs, Targets}).

get_output_buf(#nninput{ ibuf=Buf, icount=Count }) ->
    {Buf, Count};
get_output_buf(#nnlayer{ obuf=Buf, ocount=Count }) ->
    {Buf, Count}.

-define(SOURCE, "

static float sigmoid(float x)
{
     float exp_value;
     float return_value;

     exp_value = exp(-x);

     return_value = 1 / (1 + exp_value);

     return return_value;
}

__kernel void nnforward(__global __read_only  float *inputs,
                        __global __write_only float *output,
                        __global __read_only  float *weights,
                        const int input_count,
                        const int output_count) {

  int id = get_global_id(0);
  if (id >= output_count) return;

  float dp = 0;
  int i = 0;

  for( ; i < input_count-3 ; i += 4) {
    float *in = (float*) (inputs  + i);
    float *w  = (float*) (weights + id*(input_count+1) + i);
    dp = dp + dot( *(float4*)in, *(float4*)w );
  }

  for (; i < input_count; i++) {
    float *in = (float*) (inputs  + i);
    float *w  = (float*) (weights + id*(input_count+1) + i);
    dp += *in * *w;
  }

  output[id] = sigmoid( dp + weights[ id *(input_count+1) + input_count ] );
}

__kernel void nnbackprop(__global __read_only  float *outputs,
                         __global __read_only  float *target_deltas,
                         __global __read_only  float *target_weights,
                         __global __write_only float *deltas,
                         const int count,
                         const int target_count)
{
  int j = get_global_id(0);
  if (j >= count) return;

  float dp = 0;
  int q = 0;
  for (; q < target_count; q++) {
    float *td = (float*) (target_deltas  + q);
    float *tw = (float*) (target_weights + q*(count+1) + j);
    dp += *td * *tw;
  }

  deltas[j] = outputs[j] * (1 - outputs[j]) * dp;
}

__kernel void nnadjust(__global __read_only  float *inputs,
                       __global __read_only  float *deltas,
                       __global __read_write float *weights,
                       const int gamma,
                       const int input_count,
                       const int output_count) {
  int j = get_global_id(0);
  if (j >= output_count) return;

  for (int i = 0; i < input_count+1; i++) {
    float adj = -gamma * inputs[i] * deltas[j];
    float *w  = (float*) (weights + j*(input_count+1) + i);
    *w += adj;
  }
}



").

init([Inputs,Layers]) ->
    E = clu:setup(gpu),
    {ok, Program} = clu:build_source(E, ?SOURCE),
    {ok, Q} = cl:create_queue(E#cl.context, clu:device(E), []),
    {ok, InputBuf}  = cl:create_buffer(E#cl.context, [read_write], Inputs * 4),
    RevPipeLine = lists:foldl( fun( OCount, Acc) ->
                                       {IBuf,ICount} = get_output_buf(hd(Acc)),
                                       {ok, OBuf}    = cl:create_buffer(E#cl.context, [read_write], (OCount+1) * 4),
                                       {ok, _}       = cl:enqueue_write_buffer(Q, OBuf, OCount*4, 4, <<1.0:32/native-float>>, []),
                                       {ok, DBuf}    = cl:create_buffer(E#cl.context, [read_write], OCount * 4),
                                       WCount        = (ICount+1) * OCount,
                                       {ok, WBuf}    = cl:create_buffer(E#cl.context, [read_write], WCount * 4),

                                       {ok, Forward}  = cl:create_kernel(Program, "nnforward"),
                                       {ok, Backprop}  = cl:create_kernel(Program, "nnbackprop"),
                                       {ok, Adjust}  = cl:create_kernel(Program, "nnadjust"),
                                       {ok, Local} = cl:get_kernel_workgroup_info(Forward, clu:device(E), work_group_size),
                                       ok = clu:apply_kernel_args(Forward, [IBuf, OBuf, WBuf, ICount, OCount]),
                                       ok = clu:apply_kernel_args(Adjust, [IBuf, DBuf, WBuf, ?GAMMA, ICount, OCount]),

                                       [#nnlayer{ ibuf=IBuf, icount=ICount,
                                                  obuf=OBuf, ocount=OCount,
                                                  wbuf=WBuf, wcount=WCount,
                                                  dbuf=DBuf,
                                                  kernel=Forward,
                                                  backprop=Backprop,
                                                  adjust=Adjust,
                                                  max_local=Local }|Acc]
                               end,
                               [#nninput{ ibuf=InputBuf, icount=Inputs }],
                               Layers
                             ),
    [In|Inners] = lists:reverse(RevPipeLine),

    %% setup training pipeline
    [ #nnlayer{ dbuf=ODBuf, wbuf=OWBuf, ocount=OCount } | Rev ] = lists:reverse(Inners),
    lists:foldl( fun( #nnlayer{ backprop=Backprop, obuf=OBuf, dbuf=DBuf, ocount=Count }, {TDBuf, TWBuf, TCount}) ->
                         ok = clu:apply_kernel_args(Backprop, [OBuf, TDBuf, TWBuf, DBuf, Count, TCount])
                 end,
                 { ODBuf, OWBuf, OCount },
                 Rev ),

    {ok, #state{ e=E, input=In, layers=Inners, queue=Q }}.

handle_call(randomize_weights, _From, State=#state{ layers=Layers, queue=Q }) ->
    Events = lists:map( fun(#nnlayer{ wbuf=WBuf, wcount=WCount }) ->
                                Random = [ random:uniform() || _ <- lists:seq(1, WCount) ],
                                Data = << <<Value:32/native-float>> || Value <- Random  >>,
                                {ok, E0} = cl:enqueue_write_buffer(Q, WBuf, 0, byte_size(Data), Data, []),
                                E0
                        end,
                        Layers ),
    cl:enqueue_wait_for_events(Q, Events),
    {reply, ok, State};

handle_call({execute, InData}, _From, State=#state{ input=#nninput{ ibuf=IMem, icount=ICount }, layers=Layers, queue=Q }) ->
    case pack_inputs(InData, ICount) of
        {error, _} = Error ->
            {reply, Error, State};

        {ok, InputBin} ->
            {ok, E0} = cl:enqueue_write_buffer(Q, IMem, 0, byte_size(InputBin), InputBin, []),
            {ExecDone, OMem, OSize} =
                lists:foldl(fun( #nnlayer{ kernel=Kernel, ocount=OCount, max_local=Local, obuf=OMem },
                                {E1, _, _}) ->
                                   {ok, E2} = cl:enqueue_nd_range_kernel(Q,
                                                                         Kernel,
                                                                         [OCount],
                                                                         [min(Local,OCount)],
                                                                         [E1]),
                                   {E2, OMem, OCount}
                           end,
                           {E0, IMem, ICount},
                           Layers),
            {ok, ReadEvent} = cl:enqueue_read_buffer(Q, OMem, 0, 4*OSize, [ExecDone]),
            {ok, OutBin} = cl:wait(ReadEvent),
            Result = [ Num || <<Num:32/native-float>> <= OutBin ],
            {reply, {ok, Result}, State}
    end;

%% we got O, we want T
handle_call({correct, O, T}, _From, State=#state{ layers=Layers, queue=Q }) ->

    %%
    %% Step 1: compute the deltas
    %%
    Delta = lists:map(fun({Oj,Tj}) ->
                              Oj * (1-Oj) * (Oj-Tj)
                      end,
                      lists:zip(O,T)),
    {ok, ODeltaBin} = pack_inputs(Delta, length(Delta)),
    [ #nnlayer{ dbuf=ODBuf } | Rev ] = lists:reverse(Layers),
    {ok, E0} = cl:enqueue_write_buffer(Q, ODBuf, 0, byte_size(ODeltaBin), ODeltaBin, []),
    E3 = lists:foldl( fun( #nnlayer{ backprop=Backprop, max_local=Local, ocount=Count }, E1) ->
                              {ok, E2} = cl:enqueue_nd_range_kernel(Q,
                                                                    Backprop,
                                                                    [Count],
                                                                    [min(Local,Count)],
                                                                    [E1]),
                              E2
                      end,
                      E0,
                      Rev ),

    ok = cl:enqueue_wait_for_events(Q, [E3]),

    %%
    %% Step 2: Adjust weights
    %%

    Events = lists:map(fun(#nnlayer{ adjust=Adjust, max_local=Local, ocount=Count }) ->
                               {ok, E4} = cl:enqueue_nd_range_kernel(Q,
                                                                     Adjust,
                                                                     [Count],
                                                                     [min(Local,Count)],
                                                                     []),
                               E4
                       end,
                       Layers),

    ok = cl:enqueue_wait_for_events(Q, Events),

    {reply, ok, State}.


pack_inputs(InData, ICount) ->
    if
       %% input is a tuple
       is_tuple(InData) andalso size(InData) =:= ICount ->
            IList = tuple_to_list(InData),
            {ok, << <<Value:32/native-float>> || Value <- IList >>};

       %% list of floats
       is_list(InData) andalso length(InData) =:= ICount ->
            {ok, << <<Value:32/native-float>> || Value <- InData >>};

       %% assume it's a float array
       is_binary(InData) andalso byte_size(InData) =:= (ICount*4) ->
            {ok, InData};

       true ->
            {error, {badarg, InData}}
    end.

