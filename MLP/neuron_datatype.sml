datatype neuron = Neuron of value list * value

fun mkNeuron (n : int) : neuron =
  let
    fun make(0, values) = Neuron(values, mkDefaultValue (uniform ())) (* make bias *)
      | make(n, values) = make(n-1, (mkDefaultValue (uniform ()))::values)
  in
    make(n, [])
  end

(* function to calculate the output of the neuron *)
fun forwardNeuron (Neuron(neurons, weight), x) : value =
    let
        val act = add(dot(neurons, x),  weight)
        val out = tanh(act) (* maybe do sigmoid instead *)
    in
      out
    end

fun getNeuronParams(Neuron(neurons, weight)) = neurons @ [weight]
