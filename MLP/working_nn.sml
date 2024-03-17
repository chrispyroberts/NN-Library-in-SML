val globalIDCounter = ref 0

(* Define the value datatype with a record *)

datatype value = Node of value list 
               | Data of real ref 
               | Grad of real ref 
               | Backward of unit -> unit 
               | Children of value list 
               | NodeID of int


(* ALL OF THESE FUNCTIONS SHOULD GO INTO SOME LIBRARY SML FILE THAT WE GIVE TO STUDENTS *)
(*------------------------------------------ *)
(* Returns the data ref cell of a node *)
fun getData(Node a : value) : real ref =
  let

    val (Data d)::(Grad g)::(Backward b)::(Children c)::(NodeID id)::[] = a
  in
    d
  end

(* Returns the grad ref cell of a node *)
fun getGrad(Node a : value) : real ref =
  let
    val (Data d)::(Grad g)::(Backward b)::(Children c)::(NodeID id)::[] = a
  in
    g
  end 

(* Returns the backward function of a node *)
fun getBackward(Node a : value) : unit -> unit =
  let
    val (Data d)::(Grad g)::(Backward b)::(Children c)::(NodeID id)::[] = a
  in
    b
  end 

(* Returns the backward function of a node *)
fun getChildren(Node a : value) : value list =
  let
    val (Data d)::(Grad g)::(Backward b)::(Children c)::(NodeID id)::[] = a
  in
    c
  end 

(* Returns the backward function of a node *)
fun getNodeID(Node a : value) : int =
  let
    val (Data d)::(Grad g)::(Backward b)::(Children c)::(NodeID id)::[] = a
  in
    id
  end 

(* real -> value *)
fun mkDefaultValue d = 
  let
    val data = Data (ref d)
    val grad = Grad (ref 0.0)
    val backwards = Backward (fn () => ())
    val children = Children []
    val id = NodeID (globalIDCounter := !globalIDCounter + 1; !globalIDCounter)
  in
    Node [data, grad, backwards, children, id]
  end

(* real ref * real ref * (() -> ()) * value list list * int -> value list  *)
fun mkValue(d, g, b, c) =
  let
    val data = Data d
    val grad = Grad g
    val backwards = Backward b
    val children = Children c
    val id = NodeID (globalIDCounter := !globalIDCounter + 1; !globalIDCounter)
  in
    Node [data, grad, backwards, children, id]
  end


(* Check if two values are equal by comparing their unique ID *)
(* (value list * value list) -> bool *)
fun valueEq x y = (getNodeID x) = (getNodeID y)

(* --------------------------------------------- *)
(* Implemented operations:
    - addition
    - multiplication
    - tanh                      *)



(* value list * value list -> value list *)
(* Derivative:  
    d(add(x, y)) / dx = 1.0
    d(add(x, y)) / dy = 1.0

*)
fun add(x : value, y : value) =
  let
    val xData = getData(x)
    val yData = getData(y)
    val xGrad = getGrad(x)
    val yGrad = getGrad(y)

    val outData = ref (!xData + !yData)
    val outGrad = ref 0.0

    fun outBackwards () =
      let
        val dx = 1.0
        val dy = 1.0
      in
        (xGrad := !xGrad + dx * !outGrad;
         yGrad := !yGrad + dy * !outGrad)
      end

    val children = [x, y]

  in
    mkValue(outData, outGrad, outBackwards, children)
  end

(* value list * value list -> value list *)
(* Derivative:  
    subtract(x, y) = x -
    d(subtract(x, y)) / dx = 1.0
    d(subtract(x, y)) / dy = ~1.0
*)
fun subtract(x : value, y : value) =
  let
    val xData = getData(x)
    val yData = getData(y)
    val xGrad = getGrad(x)
    val yGrad = getGrad(y)

    val outData = ref (!xData - !yData)
    val outGrad = ref 0.0

    fun outBackwards () =
      let
        val dx = 1.0
        val dy = ~1.0
      in
        (xGrad := !xGrad + dx * !outGrad;
         yGrad := !yGrad + dy * !outGrad)
      end

    val children = [x, y]
  in
    mkValue(outData, outGrad, outBackwards, children)
  end


(* value list * value list -> value list *)
(* Derivative:  
    d(mul(x, y)) / dx = y
    d(mul(x, y)) / dy = x

*)
fun mul(x, y) =
  let
    val xData = getData(x)
    val yData = getData(y)
    val xGrad = getGrad(x)
    val yGrad = getGrad(y)

    val outData = ref (!xData * !yData)
    val outGrad = ref 0.0

    fun outBackwards () =
      let
        val dx = !yData
        val dy = !xData
      in
        (xGrad := !xGrad + dx * !outGrad;
         yGrad := !yGrad + dy * !outGrad)
      end
    
    val children = [x, y]
  in
    mkValue(outData, outGrad, outBackwards, children)
  end

(* value list * value list -> value list *)
(* Derivative
    d(tanH(x))/dx = 1 - tanh(x)^2  *
*)
fun tanh (x) =
    let
        val xData = getData(x)
        val xGrad = getGrad(x)

        val exp2x = Math.exp(2.0 * !xData)
        val outData = ref ((exp2x - 1.0) / (exp2x + 1.0))
        val outGrad = ref 0.0

        fun outBackward () =
          let
            val dx = 1.0 - (!outData)*(!outData)
          in
            (xGrad := !xGrad +  dx * !outGrad)
          end

        val children = [x]
    in
        mkValue (outData, outGrad, outBackward, children)
    end

(* -------------------------------------------------------- *)
(* Function to build the list of nodes in topological order *)
fun buildTopo (n, visited) =
  if List.exists (valueEq n) visited
  then (visited, [])
  else

    let
        val newVisited = n :: visited

        fun searchChildren (child, (vis, topo)) =
            let
                val (newVis, newTopo) = buildTopo (child, vis)
            in
                (newVis, newTopo @ topo)
            end

        val (finalVisited, childrenTopo) = foldl searchChildren (newVisited, []) (getChildren n)
    in
      (finalVisited, childrenTopo @ [n])
    end


(* Backwards function *)
fun backwards (x) =
    let
        fun applyBackward (_, []) = ()
          | applyBackward (startNode, nodes) = 
              (getGrad(startNode) := 1.0;
              foldr (fn (n, ()) => getBackward(n) ()) () nodes)

        val (_, topo) = buildTopo (x, [])
    in
        applyBackward(x, topo)
    end

(* -------------------------------------------------------- *)

(* now to make multilayer perceptron *)

val randomCount = ref 1

fun uniform () =
    let
      val r = (randomCount := !randomCount+1; Random.rand(0, !randomCount))
      val sign = Random.randRange (0, 1) r
      val rand = Random.randReal r
    in
      case sign of
           0 => rand * ~1.0
         | _ => rand
    end


datatype neuron = Neuron of value list * value


fun mkNeuron (n : int) : neuron =
  let
    fun make(0, values) = Neuron(values, mkDefaultValue (uniform ())) (* make bias *)
      | make(n, values) = make(n-1, (mkDefaultValue (uniform ()))::values)
  in
    make(n, [])
  end


exception IncompatibleSizes

(* A dot product sum operator for values *)
fun dot ([], []) = (mkDefaultValue 0.0)
  | dot (w::ws, x::xs) = add(mul(w, x), dot (ws, xs))
  | dot (_, _) = raise IncompatibleSizes

(* function to convert some list of reals to values *)
fun toValues [] = []
  | toValues (x::xs) = (mkDefaultValue x) :: (toValues xs)

(* function to calculate the output of the neuron *)
fun forwardNeuron (Neuron(neurons, weight), x) : value =
    let
        val act = add(dot(neurons, x),  weight)
        val out = tanh(act) (* maybe do sigmoid instead *)
    in
      out
    end


fun getNeuronParams(Neuron(neurons, weight)) = neurons @ [weight]


(* testing 
val x = toValues [2.0, 3.0]
val n = mkNeuron(2)
val res = forwardNeuron(n, x)
printData(res);
*)


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

(* loss : value list * value list -> value *)
fun loss ([], []) = mkDefaultValue(0.0)
  | loss (pred::xs, y::ys) = add(mul(subtract(pred, y), subtract(pred, y)), loss(xs, ys))

(* zip helper! *)
fun zip ([], []) = []
  | zip (x::xs, y::ys) = (x, y) :: zip(xs, ys)

(* helper! *)
(* value list -> real list *)
fun toList vs = List.map (fn v => !(getData v)) vs

(* train : mpl * value list list * value list * int * real ->  real list * unit list *)
fun train(model, data, target, numEpochs, lr) =
  let
    fun flatten l = foldl op@ [] l
    
    val parameters = getMLPParams(model)
    fun resetGradients () = foldl (fn (n, ()) => ((getGrad n) := 0.0)) () parameters
    
    fun updateParams() : unit = 
      let
        fun update(n : value) : unit = (getData n := !(getData n) - lr * !(getGrad n))
      in
        List.foldl (fn (n, ()) => update(n)) () parameters
      end
        
    fun trainHelp(model, 0, currEpoch, final_prediction, epochStatList) = (toList final_prediction, epochStatList)
      | trainHelp(model, n, currEpoch, _, epochStatList) =
      let
        val y_pred = flatten (List.map (fn x => forwardMLP(model, x)) data)
        val l = loss(y_pred, target)
        
        (* reset gradients of all parameters to 0 before calling backwards pass*)
        val () = resetGradients ()
        
        (* update gradients *)
        val () = backwards(l) 

        (* update params *)
        val () = updateParams ()

        (* Save stats *)
        val stats = (fn () => print("Epoch: " ^ Int.toString currEpoch ^ 
                                   " Loss: " ^ (Real.toString (!(getData l))) 
                                   ^ "\n"))
      in
        trainHelp(model, n-1, currEpoch+1, y_pred, stats :: epochStatList)
      end
  in
    trainHelp(model, numEpochs, 0, [], [])
  end

(* END OF NN LIBRARY *)
val xs = List.map (toValues) [[2.0, 3.0, ~1.0],
                              [3.0, ~1.0, 0.5],
                              [0.5, 1.0, 1.0],
                              [1.0, 1.0, ~1.0]]


val ys = toValues [1.0, ~1.0, ~1.0, 1.0] (* desired targets *) 

val randomCount = ref 10

val model = mkMLP(3, [4, 4, 1])
val epochs = 20
val lr = 0.1

(* train the model! *)
val (preds, stats) = train(model, xs, ys, epochs, lr)


(* Printing stuff --------------------- *)
fun printData(Node a) = 
  let
    val (Data d) :: (Grad g) :: xs = a
  in
    print("Data: " ^ Real.toString(!d) ^ " Gradient: " ^ Real.toString(!g) ^ "\n")
  end

(* print train stats *)
fun printUnitList xs = List.foldr (fn (f, ()) => f ()) () xs

(* print predictions *)
fun printList xs = List.foldl (fn (x, ()) => print(Real.toString x ^ " ")) () xs

fun print2DList XS =
  let
    fun convert_2d_real_list_to_string xs = List.map (fn row => (List.foldl (fn (x, s)
      => s ^ " " ^  Real.toString x) " " row)) xs
  in
    print((List.foldl (fn (x, s) => s ^ "\n" ^ x) ("\n" ^ "Dataset:" ^ "\n") XS) ^ "\n" ^ "\n")
  end

val () = Control.Print.printDepth := 30;

