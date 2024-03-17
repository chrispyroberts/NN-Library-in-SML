# Building a Neural Net Library (Multi-Layer Perceptron) in SML (A Functional Programming Language)

This project will heavily rely on the ```value``` datatype and an automatic differentiation algorithnm that I implemented in [this repo](https://github.com/chrispyroberts/Auto-Diff-in-SML). If you are not familiar with backpropogation, or don't understand how automatic differentiation works, I recommend you read through the linked repo before continuing.

With that being said, we start by defining a new datatype, the ```neuron``` and a function to create a neuron. We take in some ```n```, which represents the number of inputs to this neuron, 


For example, this neuron would have 3 inputs ([_source_](https://cs231n.github.io/neural-networks-1/)):
![image](https://github.com/chrispyroberts/NN-Library-in-SML/assets/98184754/1f8a9109-4720-43fc-8e62-09cb3f921dad)

The neuron will make a weight (random value between -1.0 and 1.0) for each input, and a bias which controls how trigger happy this neuron will be. 

```SML
datatype neuron = Neuron of value list * value

fun mkNeuron (n : int) : neuron =
  let
    fun make(0, values) = Neuron(values, mkDefaultValue (uniform ())) (* make bias *)
      | make(n, values) = make(n-1, (mkDefaultValue (uniform ()))::values)
  in
    make(n, [])
  end
```
To call this neuron on some input vector ```x```, we want to calculate the dot product of the weights ```w``` with the input ```x``` to get the output ```y```
$$y = \mathbf{w\cdot x}+ b$$
After calculating this dot product, we want to pass this through an activation function. We'll use tanh as our activation function.
<p align="center">
  <img src="https://github.com/chrispyroberts/NN-Library-in-SML/assets/98184754/8db1ad89-44a8-486b-ade0-e30d47df6ed5">
</p>

We'll also make a function that returns a list of the neuron's parameters (weights and bias), which will be useful later for the backwards pass.

```SML
(* dot : value list * value list -> value *)
fun dot ([], []) = (mkDefaultValue 0.0)
  | dot (w::ws, x::xs) = add(mul(w, x), sum (ws, xs))
  | dot (_, _) = raise IncompatibleSizes

(* callNeuron : neuron * value list -> value *)
fun forwardNeuron (Neuron(neurons, weight), x) : value =
    let
        val dot = add(sum(neurons, x),  weight)
        val act = tanh(dot) 
    in
      act
    end

fun getNeuronParams(Neuron(neurons, weight)) = neurons @ [weight]
```
Now everytime we make a new neuron, it will be created with random start weights, allowing us to forward a single neuron!
```
val x = toValues [2.0, 3.0]
val n = mkNeuron(2)
val res = forwardNeuron(n, x)

printData(res);
- Data: 0.995889110975 Gradient: 0.0
```


Now we are ready to define a layer of neurons. The following is a schematic for an MLP. Each layer has a number of neurons that aren't connected to each other, but fully connected to the inputs. So a layer is a set of neurons evaluated independently. 
![image](https://github.com/chrispyroberts/NN-Library-in-SML/assets/98184754/c3e52b5b-84dd-42ef-8195-7a8752b7e994)
[_source_](https://www.google.com/url?sa=i&url=https%3A%2F%2Fcs231n.github.io%2Fconvolutional-networks%2F&psig=AOvVaw1B_BaKk6xSWUrVt1jqfSak&ust=1710720817969000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCJCKueqB-oQDFQAAAAAdAAAAABAD)

How many neurons we want in our layer we take as an input. We initialize this number of independent neurons and during the forward pass, individually evaluate them.

```SML
datatype layer = Layer of neuron list

(* mkLayer : int * int -> layer *)
fun mkLayer(nIn, nOut) =
  let
    fun make (nIn, 0, acc) = Layer acc
      | make (nIn, nOut, acc) = mkLayer(nIn, nOut-1, (mkNeuron(nIn, [])) :: acc)
  in
    make(nIn, nOut, [])
  end

(* forwardLayer : layer * value list -> value list *)
fun forwardLayer (Layer([]), x) = []
  | forwardLayer (Layer(n::neurons), x) = (callNeuron(n, x)) :: (forwardLayer(Layer neurons, x))

(* getLayerParams : layer -> value list *)
fun getLayerParams(Layer l) = foldl (fn (n, p) => p @ (getNeuronParams(n))) [] l
```

Now instead of neurons, we can make a layer of neurons.

```SML
val x = toValues [2.0, 3.0]
val l = mkLayer(2, 3)  (* Create three 2d neurons *)
val res = forwardLayer(l, x)

List.map printData res;
- Data: 0.500927989224 Gradient: 0.0
- Data: 0.517670079787 Gradient: 0.0
- Data: 0.995889110975 Gradient: 0.0
```

We can see that we have 3 independent neurons.

Finally let's complete the picture and define a Multi-layer perceptron (MLP). In an MLP, the layers just feed into each other sequentially. 

So an MLP is similar, we take a number of inputs, but instead of a single nOut, we take a list of nOuts, which defines the sizes of all the layers in the MLP. We put them all together, then iterate over all consecutive pairs of sizes, then in the forward pass, we just call them sequentially.

```SML
datatype mlp = MLP of layer list

(* mkMLP : int * int list -> mlp *)
fun mkMLP (nIn, nOuts) =
  let
    fun make (nIn, n1::[], []) = MLP([mkLayer(nIn, nOut)]) (* special case of 0 hidden layer MLP*)
      | make (0, n1::n2::[], acc) = MLP(acc @ [mkLayer(n1, n2)])
      | make (0, n1::n2::nOuts, acc) = make(0, n2::nOuts, acc @ [mkLayer(n1, n2)])
      | make (nIn, n::nOuts, acc) = make(0, n::nOuts, mkLayer(nIn, n) :: acc)
  in
    make(nIn, nOut, [])
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
```

Let's now recreate the actual picture. So we want 3 input neurons, 2 hidden layers with 4 neurons each, and an output unit.

```SML
val x = toValues [2.0, 3.0, ~1.0]
val n = mkMLP(3, [4, 4, 1]) 
val [res] = forwardMLP(n, x)

printData res;
- Data: 0.611611026853 Gradient: 0.0
```

These expressions are now getting relatively involved. Here's the computational graph of the code we just ran. This is an entire MLP that we have defined, all the way until a single output.
![image](https://github.com/chrispyroberts/NN-Library-in-SML/assets/98184754/ca06a0e6-a777-41a2-add0-fca0c542b667)

Differentiating this by hand would be a nightmare. This is why we used the ```value``` datatype, as it has an automatic differentiation feature for us!

Let's use a new dataset, with 4 examples, so 4 possible inputs to the neural network, and 4 desired targets.
```SML
val xs = List.map toValues [[2.0, 3.0, ~1.0],
                            [3.0, ~1.0, 0.5],
                            [0.5, 1.0, 1.0],
                            [1.0, 1.0, ~1.0]]

val ys = toValues [1.0, ~1.0, ~1.0, 1.0] (* desired targets *) 
```
Since our outputs are ```~1.0``` and ```1.0```, this is a simple binary classifier neural network that we'd like to build.

Let's see what our neural network thinks of this data right now.

```SML
(* the result of mapping forwardMLP is a list of singleton value lists since our output dimension is 1 so we extract the value *)
val y_pred : value list = List.map (fn [x] => x) (List.map (fn x => forwardMLP(n, x)) xs)

List.map printData y_pred;
- Data: 0.611611026853 Gradient: 0.0
- Data: 0.459266801113 Gradient: 0.0
- Data: 0.797254447499 Gradient: 0.0
- Data: 0.834587495667 Gradient: 0.0
```
We can see that we need to push the first value up, the second value down, the third value down, and the fourth value up in order to get closer to our target.

So how do we tune the weights to better predict the desired values? The trick used in deep learning to achieve this is to calculate a single number which measures the total performance of the neural network: the loss. We'll use the mean squared error loss:
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

```SML
(* Our MSE loss is defined for neural networks with a single output node *)
(* loss : value list * value list -> value *)
fun loss ([], []) = mkDefaultValue(0, 0)
  | loss (pred::xs, y::ys) = add(mul(add(pred, y), add(pred, y)), loss(xs, y))
```
Let's check what the loss of each prediction is.

```sml
val individual_l = List.map (fn (x, y) => loss([x], [y])) (zip(y_pred, ys))

List.map printData individual_l;
- Data: 2.59729010187 Gradient: 0.0
- Data: 0.292392392379 Gradient: 0.0
- Data: 0.0411057590588 Gradient: 0.0
- Data: 3.36571127926 Gradient: 0.0
```

Now of the whole network
```
val l = loss(y_pred, ys)

printData l;
- Data: 6.29649953257 Gradient: 0.0
```

