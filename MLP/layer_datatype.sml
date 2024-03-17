datatype layer = Layer of neuron list

(* function to make a layer using an accumulator *)

(* mkLayer : int * int -> layer *)
fun mkLayer(nIn, nOut) =
  let
    fun make (nIn, 0, acc) = Layer acc
      | make (nIn, nOut, acc) = make(nIn, nOut-1, (mkNeuron(nIn)) :: acc)
  in
    make(nIn, nOut, [])
  end

(* forwardLayer : layer * value list -> value list *)
fun forwardLayer (Layer([]), x) = []
  | forwardLayer (Layer(n::neurons), x) = (forwardNeuron(n, x)) :: (forwardLayer(Layer neurons, x))

(* getLayerParams : layer -> value list *)
fun getLayerParams(Layer l) = foldl (fn (n, p) => p @ (getNeuronParams(n))) [] l