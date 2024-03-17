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
