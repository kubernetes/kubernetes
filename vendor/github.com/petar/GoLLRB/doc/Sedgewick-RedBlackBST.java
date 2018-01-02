public class RedBlackBST<Key extends Comparable<Key>, Value> 
{
   private static final int BST = 0;
   private static final int TD234 = 1;
   private static final int BU23 = 2;
   private static final boolean RED   = true;
   private static final boolean BLACK = false;

   private Node root;            // root of the BST
   private int k;                // ordinal for drawing
   private final int species;    // species kind of tree for insert
   private int heightBLACK;      // black height of tree 

   RedBlackBST(int species)
   {  this.species = species;  }

   private class Node
   {
      Key key;                  // key
      Value value;              // associated data
      Node left, right;         // left and right subtrees
      boolean color;            // color of parent link
      private int N;            // number of nodes in tree rooted here
      private int height;       // height of tree rooted here
      private double xc, yc;    // for drawing

      Node(Key key, Value value)
      {
         this.key   = key;
         this.value = value;
         this.color = RED;
         this.N = 1;
         this.height = 1;
      }
   }

   public int size()
   {  return size(root);  }

   private int size(Node x)
   { 
      if (x == null) return 0;
      else           return x.N;
   }

   public int rootRank()
   { 
       if (root == null) return 0;
       else              return size(root.left);
   }

   public int height()
   {  return height(root);  }

   public int heightB()
   {  return heightBLACK;  }

   private int height(Node x)
   { 
      if (x == null) return 0;
      else           return x.height;
   }

   public boolean contains(Key key)
   {  return (get(key) != null);  }

   public Value get(Key key)
   {  return get(root, key);  }

   private Value get(Node x, Key key)
   {
      if (x == null)        return null;
      if (eq  (key, x.key)) return x.value;
      if (less(key, x.key)) return get(x.left,  key);
      else                  return get(x.right, key);
   }

   public Key min()
   {  
      if (root == null) return null;
      else              return min(root);
   }

   private Key min(Node x)
   {
      if (x.left == null) return x.key;
      else                return min(x.left);
   }

   public Key max()
   {  
      if (root == null) return null;
      else              return max(root);
   }

   private Key max(Node x)
   {
      if (x.right == null) return x.key;
      else                 return max(x.right);
   }

   public void put(Key key, Value value)
   {
      root = insert(root, key, value);
      if (isRed(root)) heightBLACK++;
      root.color = BLACK;
   }

   private Node insert(Node h, Key key, Value value)
   { 
      if (h == null) 
         return new Node(key, value);

      if (species == TD234) 
         if (isRed(h.left) && isRed(h.right))
	     colorFlip(h);

      if (eq(key, h.key))
         h.value = value;
      else if (less(key, h.key)) 
         h.left = insert(h.left, key, value); 
      else 
         h.right = insert(h.right, key, value); 

      if (species == BST) return setN(h);

      if (isRed(h.right))
         h = rotateLeft(h);

      if (isRed(h.left) && isRed(h.left.left))
         h = rotateRight(h);

      if (species == BU23)
         if (isRed(h.left) && isRed(h.right))
	     colorFlip(h);

      return setN(h);
   }

   public void deleteMin()
   {
      root = deleteMin(root);
      root.color = BLACK;
   }

   private Node deleteMin(Node h)
   { 
      if (h.left == null)
         return null;

      if (!isRed(h.left) && !isRed(h.left.left))
         h = moveRedLeft(h);

      h.left = deleteMin(h.left);

      return fixUp(h);
   }

   public void deleteMax()
   {
      root = deleteMax(root);
      root.color = BLACK;
   }

   private Node deleteMax(Node h)
   { 
       //      if (h.right == null)
	  //      {  
	  //         if (h.left != null)
	  //            h.left.color = BLACK;
	  //         return h.left;
	  //      }

      if (isRed(h.left))
         h = rotateRight(h);

      if (h.right == null)
         return null;

      if (!isRed(h.right) && !isRed(h.right.left))
         h = moveRedRight(h);

      h.right = deleteMax(h.right);

      return fixUp(h);
   }

   public void delete(Key key)
   { 
      root = delete(root, key);
      root.color = BLACK;
   }

   private Node delete(Node h, Key key)
   { 
      if (less(key, h.key)) 
      {
         if (!isRed(h.left) && !isRed(h.left.left))
            h = moveRedLeft(h);
         h.left =  delete(h.left, key);
      }
      else 
      {
         if (isRed(h.left)) 
            h = rotateRight(h);
         if (eq(key, h.key) && (h.right == null))
            return null;
         if (!isRed(h.right) && !isRed(h.right.left))
            h = moveRedRight(h);
         if (eq(key, h.key))
         {
            h.value = get(h.right, min(h.right));
            h.key = min(h.right);
	    h.right = deleteMin(h.right);
         }
         else h.right = delete(h.right, key);
      }
 
      return fixUp(h);
   }

// Helper methods

    private boolean less(Key a, Key b) { return a.compareTo(b) <  0; }
    private boolean eq  (Key a, Key b) { return a.compareTo(b) == 0; }

    private boolean isRed(Node x)
    {
        if (x == null) return false;
        return (x.color == RED);
    }

   private void colorFlip(Node h)
   {  
      h.color        = !h.color;
      h.left.color   = !h.left.color;
      h.right.color  = !h.right.color;
   }

   private Node rotateLeft(Node h)
   {  // Make a right-leaning 3-node lean to the left.
      Node x = h.right;
      h.right = x.left;
      x.left = setN(h);
      x.color      = x.left.color;                   
      x.left.color = RED;                     
      return setN(x);
   }

   private Node rotateRight(Node h)
   {  // Make a left-leaning 3-node lean to the right.
      Node x = h.left;
      h.left = x.right;
      x.right = setN(h);
      x.color       = x.right.color;                   
      x.right.color = RED;                     
      return setN(x);
    }

    private Node moveRedLeft(Node h)
    {  // Assuming that h is red and both h.left and h.left.left
       // are black, make h.left or one of its children red.
       colorFlip(h);
       if (isRed(h.right.left))
       { 
          h.right = rotateRight(h.right);
          h = rotateLeft(h);
          colorFlip(h);
       }
      return h;
    }

    private Node moveRedRight(Node h)
    {  // Assuming that h is red and both h.right and h.right.left
       // are black, make h.right or one of its children red.
       colorFlip(h);
       if (isRed(h.left.left))
       { 
          h = rotateRight(h);
	  colorFlip(h);
       }
       return h;
    }

   private Node fixUp(Node h)
   {
      if (isRed(h.right))
         h = rotateLeft(h);

      if (isRed(h.left) && isRed(h.left.left))
         h = rotateRight(h);

      if (isRed(h.left) && isRed(h.right))
         colorFlip(h);

      return setN(h);
   }
      
   private Node setN(Node h)
   {
      h.N = size(h.left) + size(h.right) + 1;
      if (height(h.left) > height(h.right)) h.height = height(h.left) + 1;
      else                                  h.height = height(h.right) + 1;
      return h;
   }
      
   public String toString()
   {  
      if (root == null) return "";
      else              return heightB() + " " + toString(root);  
   }

   public String toString(Node x)
   {  
      String s = "(";
      if (x.left == null) s += "("; else s += toString(x.left);
      if (isRed(x)) s += "*";
      if (x.right == null) s += ")"; else s += toString(x.right);
      return s + ")";
   }

// Methods for tree drawing

    public void draw(double y, double lineWidth, double nodeSize)
    { 
      k = 0; 
      setcoords(root, y);
      StdDraw.setPenColor(StdDraw.BLACK);
      StdDraw.setPenRadius(lineWidth);
      drawlines(root); 
      StdDraw.setPenColor(StdDraw.WHITE);
      drawnodes(root, nodeSize); 
    }

    public void setcoords(Node x, double d)
    {
        if (x == null) return;
        setcoords(x.left, d-.04);
        x.xc = (0.5 + k++)/size(); x.yc = d - .04;
        setcoords(x.right, d-.04);
    }

    public void drawlines(Node x)
    {
        if (x == null) return;
        drawlines(x.left);
        if (x.left != null)
        {
	    if (x.left.color == RED) StdDraw.setPenColor(StdDraw.RED);
	    else StdDraw.setPenColor(StdDraw.BLACK);
            StdDraw.line(x.xc, x.yc, x.left.xc, x.left.yc);
        }
        if (x.right != null)
        {
	    if (x.right.color == RED) StdDraw.setPenColor(StdDraw.RED);
	    else StdDraw.setPenColor(StdDraw.BLACK);
            StdDraw.line(x.xc, x.yc, x.right.xc, x.right.yc);
        }
        drawlines(x.right);
    }

    public void drawnodes(Node x, double nodeSize)
    {
        if (x == null) return;
        drawnodes(x.left, nodeSize);
	StdDraw.filledCircle(x.xc, x.yc, nodeSize);
        drawnodes(x.right, nodeSize);
    }

    public void mark(Key key)
    { 
      StdDraw.setPenColor(StdDraw.BLACK);
      marknodes(key, root); 
    }

    public void marknodes(Key key, Node x)
    {
        if (x == null) return;
        marknodes(key, x.left);
	if (eq(key, x.key))
           StdDraw.filledCircle(x.xc, x.yc, .004);
        marknodes(key, x.right);
    }

    public int ipl()
    {  return ipl(root);  }

    public int ipl(Node x)
    {
	if (x == null) return 0;
	return size(x) - 1 + ipl(x.left) + ipl(x.right);
    }
       
    public int sizeRed()
    {  return sizeRed(root);  }

    public int sizeRed(Node x)
    {
	if (x == null) return 0;
        if (isRed(x)) return 1 + sizeRed(x.left) + sizeRed(x.right);
        else          return sizeRed(x.left) + sizeRed(x.right);
    }
       
// Integrity checks

   public boolean check() 
   {  // Is this tree a red-black tree?
      return isBST() && is234() && isBalanced();
   }

   private boolean isBST() 
   {  // Is this tree a BST?
      return isBST(root, min(), max());
   }

   private boolean isBST(Node x, Key min, Key max)
   {  // Are all the values in the BST rooted at x between min and max,
      // and does the same property hold for both subtrees?
      if (x == null) return true;
      if (less(x.key, min) || less(max, x.key)) return false;
      return isBST(x.left, min, x.key) && isBST(x.right, x.key, max);
   } 

   private boolean is234() { return is234(root); }
   private boolean is234(Node x)
   {  // Does the tree have no red right links, and at most two (left)
      // red links in a row on any path?
      if (x == null) return true;
      if (isRed(x.right)) return false;
      if (isRed(x))
        if (isRed(x.left))
          if (isRed(x.left.left)) return false;
      return is234(x.left) && is234(x.right);
   } 

   private boolean isBalanced()
   { // Do all paths from root to leaf have same number of black edges?
      int black = 0;     // number of black links on path from root to min
      Node x = root;
      while (x != null)
      {
         if (!isRed(x)) black++;
            x = x.left;
      }
      return isBalanced(root, black);
   }

   private boolean isBalanced(Node x, int black)
   { // Does every path from the root to a leaf have the given number 
     // of black links?
      if      (x == null && black == 0) return true;
      else if (x == null && black != 0) return false;
      if (!isRed(x)) black--;
      return isBalanced(x.left, black) && isBalanced(x.right, black);
   } 


  public static void main(String[] args)
  {
      StdDraw.setPenRadius(.0025);
      int species = Integer.parseInt(args[0]);
      RedBlackBST<Integer, Integer> st;
      st = new RedBlackBST<Integer, Integer>(species);
      int[] a = { 3, 1, 4, 2, 5, 9, 6, 8, 7 };
      for (int i = 0; i < a.length; i++)
	  st.put(a[i], i);
      StdOut.println(st);
      StdDraw.clear(StdDraw.LIGHT_GRAY);
      st.draw(.95, .0025, .008);
      StdOut.println(st.min() + " " + st.max() + " " + st.check());
      StdOut.println(st.ipl());
      StdOut.println(st.heightB());
  }

}
