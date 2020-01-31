
import java.util.ArrayList;

class NeuralNetwork {
    int input, hidden, output, noOfHiddenLayers;
    float learningRate;
    boolean softMax;
    ArrayList<Matrix> weightsMatrices;
    ArrayList<Matrix> feedForwardWeights;
    ArrayList<Matrix> biasMatrices;
    //for the back propagation
    Matrix matrixMain,errors,matrixPrev,matrixBias,matrixWeights;


    Matrix weights_IH, weights_HO, bias_H, bias_O;

    NeuralNetwork ( int input, int hidden, int output,int noOfHiddenLayers) {
        softMax = true;
        this.input = input;
        this.hidden = hidden;
        this.output = output;
        this.noOfHiddenLayers = noOfHiddenLayers;

        weightsMatrices = new ArrayList<Matrix>();
        feedForwardWeights = new ArrayList<Matrix>();
        biasMatrices = new ArrayList<Matrix>();

        weightsMatrices.add(new Matrix(hidden, input));
        biasMatrices.add(new Matrix(hidden, 1));
        if (noOfHiddenLayers > 1) {   // Only if there are more than 1 hidden layers do we need to worry

        }
        weightsMatrices.add(new Matrix(output, hidden));
        biasMatrices.add(new Matrix(output, 1));
        // randomise weights
        for (Matrix weightsMatrix : weightsMatrices) {
            weightsMatrix.randomise();
        }
        for (Matrix biasMatrix : biasMatrices) {
            biasMatrix.randomise();
        }


//        weights_IH = new Matrix(hidden, input);    // weights between in and hidden
//        weights_HO = new Matrix(output, hidden);   // weights between hidden and out
//
//        weights_IH.randomise();
//        weights_HO.randomise();
//
//        bias_H = new Matrix(hidden, 1);  //weights for the bias for the hidden;
//        bias_O = new Matrix(output, 1);  //weights for the bias for the output;
//
//        bias_H.randomise();
//        bias_O.randomise();
    }
//    NeuralNetwork ( NeuralNetwork nn_) {
//        softMax = nn_.softMax;
//        input = nn_.input;
//        hidden = nn_.hidden;
//        output = nn_.output;
//
//        weights_IH = nn_.weights_IH;    // weights between in and hidden
//        weights_HO = nn_.weights_HO;  // weights between hidden and out
//
//        bias_H = nn_.bias_H;       //weights for the bias for the hidden;
//        bias_O = nn_.bias_O;      //weights for the bias for the output;
//
//    }

    static NeuralNetwork nncopy(NeuralNetwork n_) {

        NeuralNetwork nc = new NeuralNetwork(1,1,1,1);
        nc.softMax = n_.softMax;
        nc.input = n_.input;
        nc.hidden = n_.hidden;
        nc.output = n_.output;
        nc.noOfHiddenLayers = n_.noOfHiddenLayers;
        nc.learningRate = n_.learningRate;
        //TODO amend the copy logic here
        nc.weights_IH = Matrix.matrixCopy(n_.weights_IH);
        nc.weights_HO = Matrix.matrixCopy(n_.weights_HO);
        nc.bias_H = Matrix.matrixCopy(n_.bias_H);
        nc.bias_O = Matrix.matrixCopy(n_.bias_O);

        return nc;

    }

    float[] predict(float[] inArray_) {
        feedForwardWeights.clear();
        // get matrix from input array
        feedForwardWeights.add(new Matrix(inputArrayToMatrix(inArray_)));

        ////// This is our feed forward
        feedForwardWeights.add(new Matrix(calculateWeights(weightsMatrices.get(0),feedForwardWeights.get(0),biasMatrices.get(0))));
        if (noOfHiddenLayers > 1) {
            // Feed in the other layers
        }
        // Take these squished inputs and multiply them by the hidden wights
        Matrix outMatrix = new Matrix(calculateWeights(weightsMatrices.get(noOfHiddenLayers),feedForwardWeights.get(noOfHiddenLayers),biasMatrices.get(noOfHiddenLayers)));
        //////

        ///// Handle the output
        float[] ffArray = new float[output];  //  outputs as an array
        ffArray = Matrix.toArray(outMatrix);  //  create Array from outputs
        // Just need to normalise probabilities to total 1 Softmax implementation

        if (softMax) {
            float sum = 0;
            for (int i = 0; i < ffArray.length; i++) {
                ffArray[i] = (float) Math.exp((float) ffArray[i]);
                sum = sum + ffArray[i];
            }
            for (int i = 0; i < ffArray.length; i++) {
                ffArray[i] = ffArray[i] / sum;
            }

        }
        return ffArray;  // This is our predicted results Array
    }
    void train (float[] inArray_, float[] targets_, float lr_) {
        feedForwardWeights.clear();
        learningRate = lr_;
        // Get the targets as a matrix
        Matrix matrixTargets  = new Matrix(targets_.length, 1);
        matrixTargets.fromArray(targets_)  ;   // targets as a matrix
        // get matrix from input array
        feedForwardWeights.add(new Matrix(inputArrayToMatrix(inArray_)));

        ////// This is our feed forward

        feedForwardWeights.add(new Matrix(calculateWeights(weightsMatrices.get(0),feedForwardWeights.get(0),biasMatrices.get(0))));
        /////////////////Matrix hiddenM = new Matrix(calculateWeights(weightsMatrices.get(0),feedForwardWeights.get(0),biasMatrices.get(0)));
        if (noOfHiddenLayers > 1) {
            // Feed in the other layers
        }
        // Take these squished inputs and multiply them by the hidden wights
        Matrix outMatrix = new Matrix(calculateWeights(weightsMatrices.get(noOfHiddenLayers),feedForwardWeights.get(noOfHiddenLayers),biasMatrices.get(noOfHiddenLayers)));
        //////

        // This gives us our output errors Matrix.
        Matrix output_errors = new Matrix (Matrix.subMatrix(matrixTargets, outMatrix));


        // Now we have our output errors we need the gradient descent bit.
        matrixMain = outMatrix;
        errors = output_errors;
        matrixPrev = feedForwardWeights.get(noOfHiddenLayers);
        matrixBias = biasMatrices.get(noOfHiddenLayers);
        matrixWeights = weightsMatrices.get(noOfHiddenLayers);

        backPropagation(matrixMain,errors,matrixPrev,matrixBias,matrixWeights);


        // Gradient descent for the input layers
        matrixMain = feedForwardWeights.get(noOfHiddenLayers);
        errors = new Matrix(calculateErrors(weightsMatrices.get(noOfHiddenLayers),output_errors));;
        matrixPrev = feedForwardWeights.get(0);
        matrixBias = biasMatrices.get(0);
        matrixWeights = weightsMatrices.get(0);

        backPropagation(matrixMain,errors,matrixPrev,matrixBias,matrixWeights);


    }

    void backPropagation (Matrix ffMatrix, Matrix errors_Matrix,Matrix ffPrevLayer, Matrix bias, Matrix weights) {
        ffMatrix = derivative(ffMatrix,errors_Matrix);

        //update bias
        bias.addMatrix(ffMatrix);

        //calculate deltas
        Matrix weight_HO_deltas = new Matrix(calculateDeltas(ffMatrix,ffPrevLayer));

        //Adjust deltas
        weights.addMatrix(weight_HO_deltas);
    }

    Matrix calculateErrors (Matrix weights,Matrix errors) {
        // transpose Hidden to output Weights
        Matrix tmpMatrix = new Matrix(Matrix.transpose(weights));
        // get the hidden layers errors.
        Matrix tmpErrors = new Matrix(Matrix.multiply(tmpMatrix, errors));
        return tmpErrors;


    }

    void  mutate(float percentage) {


        Matrix tmp_IH = new Matrix(hidden, input);    // weights between in and hidden
        Matrix tmp_HO = new Matrix(output, hidden);   // weights between hidden and out
        tmp_HO = weights_HO;
        tmp_IH = weights_IH;
        tmp_IH.mutate(percentage);
        tmp_HO.mutate((percentage));
        weights_HO = tmp_HO;
        weights_IH = tmp_IH;

    }

    void setSoftMaxOn () {
        softMax = true;
    }
    void setSoftMaxOff () {
        softMax = false;
    }

    Matrix  calculateWeights (Matrix weights_, Matrix weightsLayer_, Matrix bias) {
        // all inputs multiplied by weights
        Matrix Tmp = new Matrix(Matrix.multiply(weights_, weightsLayer_));
        // add in the bias weights
        Tmp.addMatrix(bias);
        // activation function on all weights (squish them to 0-1)
        Tmp.sigmoid();
        return Tmp;
    }

    Matrix derivative (Matrix bckM_,Matrix errors_) {
        // Now we have our output errors we need the gradient descent bit.
        // sigmoid derivative
        bckM_.dsigmoid();
        // multiplied by the output errors
        bckM_.multiply(errors_);
        // multiplied by the learning rate
        bckM_.multiply(learningRate);
        //update bias
        return bckM_;

    }

    Matrix calculateDeltas (Matrix out_,Matrix hidden_) {
        //calculate deltas
        Matrix hidden_T = Matrix.transpose(hidden_);
        Matrix weightTmp = new Matrix(Matrix.multiply(out_, hidden_T));
        return weightTmp;
    }

//    void backPropagation (Matrix bckM_, Matrix bckMinus1_,Matrix errors_,Matrix bias_,Matrix weightsToAdjust_) {
//        // Now we have our output errors we need the gradient descent bit.
//        // sigmoid derivative
//        bckM_.dsigmoid();
//        // multiplied by the output errors
//        bckM_.multiply(errors_);
//        // multiplied by the learning rate
//        bckM_.multiply(learningRate);
//        //update bias
//        bias_.addMatrix(bckM_);
//
//        //calculate deltas
//        Matrix hidden_T = Matrix.transpose(bckMinus1_);
//        Matrix weight_HO_deltas = new Matrix(Matrix.multiply(bckM_, hidden_T));
//
//        //Adjust deltas
//        weightsToAdjust_.addMatrix(weightsToAdjust_);
//    }

    Matrix inputArrayToMatrix(float[] inA_) {
        Matrix inMatrix = new Matrix(inA_.length, 1);
        inMatrix.fromArray(inA_);   //   this is now our inputs matrix
        return inMatrix;
    }


    float resultIndex (float[] ra_) {

        float max = -100;
        float actualResult = -100;
        for (int r =0; r<ra_.length; r++) {
            if (ra_[r] > max) {
                max = ra_[r];
                actualResult= r;

            }
        }
        return actualResult;
    }
    float resultProbability (float[] ra_, boolean pcnt_ ) {

        float max = -100;
        float actualResult = -100;
        for (int r =0; r<ra_.length; r++) {
            if (ra_[r] > max) {
                max = ra_[r];


            }
        }
        if (pcnt_){
            return (java.lang.Math.round(max * 100));
        } else return max ;

    }
}