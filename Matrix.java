//implements a generic M by N matrix data structure
//eli f.



public class Matrix {

   private int rows;
   private int columns;
   private double[][] elements;
   
   //creates a matrix of zeros
   public Matrix(int rows, int columns) {
      this.rows = rows;
      this.columns = columns;
      this.elements = new double[rows][columns];  
   }
   
   // create matrix based on 2d array
   public Matrix(double[][] data) {
      rows = data.length;
      columns = data[0].length;
      this.elements = new double[rows][columns];
      for (int i = 0; i < rows; i++)
         for (int j = 0; j < columns; j++)
            this.elements[i][j] = data[i][j];
   }
   
   public Matrix clone(Matrix B) {
      return new Matrix(this.elements);
   }
   
   public int getColumns() {
      return this.columns;
   }
   
   public int getRows() {
      return this.rows;
   }
   
   //gets the array storing all the info for this matrix
   public double[][] getElements() {
      return this.elements;  
   }
   
   //TODO
   //method to get specific elements of a matrix
   
   //changes the value of this matrix to the dot product of itself and another matrix
   public void dot(Matrix B){
      double[][] BElements = B.getElements();
    	//iterating over all rows
      for(int i = 0; i < this.rows; i++){
      	//iterating over all columns in second matrix
         for(int j = 0; j < B.getColumns(); j++){
         	//iterating over columns in arg1 
            for(int k = 0; k < this.columns; k++){
               this.elements[i][j] = this.elements[i][k]*BElements[k][j];
            }
         }
      }
   }
   
   //cacluates the transpose for a matrix does not actually alter this matrix, just generates the transpose
   public Matrix T(){
      double[][] transpose = new double[this.rows][this.columns];
      for(int i = 0; i < this.rows; i++){
         for(int j = 0; j < this.columns; j++){
            transpose[j][i] = this.elements[i][j];
         }
      }
      return new Matrix(transpose);
   }
   
   //subtracts the second matrix from the first matrix
   public void minus(Matrix B){
      if(B.getRows() != this.rows || B.getColumns() != this.columns) {
         throw new IllegalArgumentException("Matrix for subtraction has incorrect size");
      }
      double[][] BElements = B.getElements();
      for( int i = 0; i < this.rows; i++){
         for( int j = 0; j < this.columns;j++){
            this.elements[i][j] = this.elements[i][j] - BElements[i][j];
         }
      }
   }
   
   public void multiply(double scalar){
      for(int i = 0; i < this.rows;i++){
         for(int j = 0; j < this.columns; j++){
            this.elements[i][j] = this.elements[i][j]*scalar;
         }
      }
   }
}