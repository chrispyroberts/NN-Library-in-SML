datatype mlp = MLP of layer list

(* mkMLP : int * int list -> mlp *)
fun mkMLP (nIn, layers) =
  let
    fun make (nIn, nOut::[], []) = MLP([mkLayer(nIn, nOut)]) (* special case of 0 hidden layer MLP*)
      | make (0, n1::nOut::[], acc) = MLP(acc @ [mkLayer(n1, nOut)])
      | make (0, n1::n2::nOuts, acc) = make(0, n2::nOuts, acc @ [mkLayer(n1, n2)])
      | make (nIn, n::nOuts, acc) = make(0, n::nOuts, mkLayer(nIn, n) :: acc)
  in
    make(nIn, layers, [])
  end

(* forwardMLP : mlp * value list -> value list *)
fun forwardMLP (MLP (l::[]), x) = forwardLayer(l, x)
  | forwardMLP (MLP (l::ls), x) =
  let
    val update = forwardLayer(l, x)
    (* sequentially feed through values to subsequent layers *)
    val res = forwardMLP(MLP ls, update)
  in
    res
  end

(* getMLPParams : mlp -> value list *)
fun getMLPParams(MLP layers) = foldl (fn (l, p) => p @ (getLayerParams(l))) [] layers