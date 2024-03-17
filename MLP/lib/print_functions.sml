fun printData(Node a) = 
  let
    val (Data d) :: (Grad g) :: xs = a
  in
    print("Data: " ^ Real.toString(!d) ^ " Gradient: " ^ Real.toString(!g) ^ "\n")
  end

fun printUnitList xs = List.foldr (fn (f, ()) => f ()) () xs

fun printList xs = List.foldl (fn (x, ()) => print(Real.toString x ^ " ")) () xs

fun print2DList XS =
  let
    fun convert_2d_real_list_to_string xs = List.map (fn row => (List.foldl (fn (x, s)
      => s ^ " " ^  Real.toString x) " " row)) xs
  in
    print((List.foldl (fn (x, s) => s ^ "\n" ^ x) ("\n" ^ "Dataset:" ^ "\n") XS) ^ "\n" ^ "\n")
  end

val () = Control.Print.printDepth := 30;
