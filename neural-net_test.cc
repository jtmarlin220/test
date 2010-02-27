#include "neural-net.h"
#include <math.h>
#include <gtest/gtest.h>
#include <iostream>

using std::cout;

// This file contains a collection of tests that check the various pieces of code you will write.
// These tests are by no means exhaustive, and you should add tests as you see fit.  The tests use
// Google's open source unittesting framework.  More information can be found here:
//
// http://code.google.com/p/googletest/
//
// By convention, we pass the expected value as the first argument of any EXPECT_EQ or ASSERT_EQ
// call.

TEST(NeuralNetTest, SigmoidPrime) {
  EXPECT_DOUBLE_EQ(0.25, NeuralNetwork::SigmoidPrime(0));
  // Make sure that we don't have numerical issues here.
  for (int i = 0; i < 1000; ++i) {
    double pos = NeuralNetwork::SigmoidPrime(i);
    double neg = NeuralNetwork::SigmoidPrime(-i);
    //EXPECT_FALSE(isnan(neg));
    //EXPECT_FALSE(isnan(pos));
    EXPECT_LE(abs(pos - neg), 0.001);
    EXPECT_GE(pos, 0.0);
    EXPECT_LE(pos, 0.25);
  }
}

TEST(NeuralNetTest, Sigmoid) {
  EXPECT_DOUBLE_EQ(0.5, NeuralNetwork::Sigmoid(0));
  EXPECT_LT(NeuralNetwork::Sigmoid(-1000), 0.1);
  EXPECT_GT(NeuralNetwork::Sigmoid(1000), 0.9);
  // Symmetry
  EXPECT_DOUBLE_EQ(NeuralNetwork::Sigmoid(-1000),
                   1 - NeuralNetwork::Sigmoid(1000));
}

//Tests a simple case of back propagation in a hidden layer network with
//  1 input, 2 hidden units, and 2 outputs.
TEST(NeuralNetTest, HiddenLayerBackprop2) {
  // Create the network
  NeuralNetwork network;
  Node * input = new Node();
  network.AddNode(input, NeuralNetwork::INPUT);
  
  Node * hidden1 = new Node();
  Node * hidden2 = new Node();
  hidden1->AddInput(input, NULL, &network);
  hidden2->AddInput(input, NULL, &network);
  network.AddNode(hidden1, NeuralNetwork::HIDDEN);
  network.AddNode(hidden2, NeuralNetwork::HIDDEN);
  
  Node * output1 = new Node();
  Node * output2 = new Node();
  output1->AddInput(hidden1, NULL, &network);
  output1->AddInput(hidden2, NULL, &network);
  output2->AddInput(hidden1, NULL, &network);
  output2->AddInput(hidden2, NULL, &network);
  network.AddNode(output1, NeuralNetwork::OUTPUT);
  network.AddNode(output2, NeuralNetwork::OUTPUT);

  // set the initial weights
  for (size_t i = 0; i < network.weights_.size(); i++)
    network.weights_[i]->value = 0;

  Input in;
  in.values.push_back(1.0);
  Target target;
  target.values.push_back(1.0);
  target.values.push_back(0.0);
  
  double lrate = 1;

  // test the feedforward
  network.FeedForward(in);
  EXPECT_EQ(0.0, hidden1->raw_value_);
  EXPECT_EQ(0.0, hidden2->raw_value_);
  EXPECT_EQ(0.0, output1->raw_value_);
  EXPECT_EQ(0.0, output2->raw_value_);
  EXPECT_EQ(0.5, hidden1->transformed_value_);
  EXPECT_EQ(0.5, hidden2->transformed_value_);
  EXPECT_EQ(0.5, output1->transformed_value_);
  EXPECT_EQ(0.5, output2->transformed_value_);

  // backprop
  network.Backprop(in, target, lrate);
  // all the weighted inputs are initially zero, so:
  // on the first output node, error = (1.0 - 0.5) = 0.5
  // on the second, error = (0.0 - 0.5) = -0.5
  double del1 = 0.5 * NeuralNetwork::SigmoidPrime(0);
  double del2 = -0.5 * NeuralNetwork::SigmoidPrime(0);
  EXPECT_DOUBLE_EQ(del1 * lrate, output1->fixed_weight_->value);
  EXPECT_DOUBLE_EQ(del1 * lrate * 0.5, output1->weights_[0]->value);
  EXPECT_DOUBLE_EQ(del1 * lrate * 0.5, output1->weights_[1]->value);
  EXPECT_DOUBLE_EQ(del2 * lrate, output2->fixed_weight_->value);
  EXPECT_DOUBLE_EQ(del2 * lrate * 0.5, output2->weights_[0]->value);
  EXPECT_DOUBLE_EQ(del2 * lrate * 0.5, output2->weights_[1]->value);
  
  //deltas at the hidden layer will be 0 because w_kj's haven't been
  //updated yet
  EXPECT_DOUBLE_EQ(0.0, hidden1->fixed_weight_->value);
  EXPECT_DOUBLE_EQ(0.0, hidden1->weights_[0]->value);
  EXPECT_DOUBLE_EQ(0.0, hidden2->fixed_weight_->value);
  EXPECT_DOUBLE_EQ(0.0, hidden2->weights_[0]->value);
  
  //----------------second iteration-------------------------------
  
  network.FeedForward(in);
  EXPECT_EQ(0.0, hidden1->raw_value_);
  EXPECT_EQ(0.0, hidden2->raw_value_);
  EXPECT_EQ(0.5, hidden1->transformed_value_);
  EXPECT_EQ(0.5, hidden2->transformed_value_);
  
  //just making sure weights are what they're supposed to be
  EXPECT_EQ(0.125, output1->fixed_weight_->value);
  EXPECT_EQ(0.0625, output1->weights_[0]->value);
  EXPECT_EQ(0.0625, output1->weights_[1]->value);
  EXPECT_EQ(-0.125, output2->fixed_weight_->value);
  EXPECT_EQ(-0.0625, output2->weights_[0]->value);
  EXPECT_EQ(-0.0625, output2->weights_[1]->value);
  
  EXPECT_EQ(0.1875, output1->raw_value_);
  EXPECT_EQ(-0.1875, output2->raw_value_);
  EXPECT_DOUBLE_EQ(NeuralNetwork::Sigmoid(0.1875), output1->transformed_value_);
  EXPECT_DOUBLE_EQ(NeuralNetwork::Sigmoid(-0.1875), output2->transformed_value_);
  
  //store weights before backpropping
  
  //backprop
  network.Backprop(in, target, lrate);
  
  double error1 = 1 - output1->transformed_value_;
  double error2 = 0 - output2->transformed_value_;
  
  double delta1 = error1 * NeuralNetwork::SigmoidPrime(output1->raw_value_);
  double delta2 = error2 * NeuralNetwork::SigmoidPrime(output2->raw_value_);
  
  EXPECT_EQ(0.125 + delta1 * lrate,
			output1->fixed_weight_->value);
  EXPECT_EQ(-0.125 + delta2 * lrate,
            output2->fixed_weight_->value);
			
  EXPECT_DOUBLE_EQ(0.0625 + delta1 * lrate * 0.5,
            output1->weights_[0]->value);
  EXPECT_DOUBLE_EQ(0.0625 + delta1 * lrate * 0.5,
            output1->weights_[1]->value);
  EXPECT_DOUBLE_EQ(-0.0625 + delta2 * lrate * 0.5,
            output2->weights_[0]->value);
  EXPECT_DOUBLE_EQ(-0.0625 + delta2 * lrate * 0.5,
            output2->weights_[1]->value);
			
  double hidden_delta_1 = NeuralNetwork::SigmoidPrime(0.0)*(.0625 * delta1 - .0625 * delta2);
  double hidden_delta_2 = NeuralNetwork::SigmoidPrime(0.0)*(.0625 * delta1 - .0625 * delta2);
  EXPECT_EQ(hidden_delta_1 * 1.0 * lrate, hidden1->weights_[0]->value);
  EXPECT_EQ(hidden_delta_1 * 1.0 * lrate, hidden1->fixed_weight_->value);
  EXPECT_EQ(hidden_delta_2 * 1.0 * lrate, hidden2->weights_[0]->value);
  EXPECT_EQ(hidden_delta_2 * 1.0 * lrate, hidden2->fixed_weight_->value);
  
}

// Tests a simple case of back propagation in a hidden layer network.
TEST(NeuralNetTest, HiddenLayerBackprop) {
  // Create a network with a single hidden unit.
  // Network:
  // INPUT --> HIDDEN --> OUTPUT
  NeuralNetwork network;
  Node* input = new Node();
  network.AddNode(input, NeuralNetwork::INPUT);
  Node* hidden = new Node();
  hidden->AddInput(input, NULL, &network);
  network.AddNode(hidden, NeuralNetwork::HIDDEN);
  Node* output = new Node();
  output->AddInput(hidden, NULL, &network);
  network.AddNode(output, NeuralNetwork::OUTPUT);
  Input in;
  in.values.push_back(1.0);
  Target target;
  target.values.push_back(1.0);
  // Initialize all weights to be 0.
  for (size_t i = 0; i < network.weights_.size(); ++i) {
    network.weights_[i]->value = 0;
  }

  double learning_rate = 0.005;
  network.FeedForward(in);

  // Expected results after feeding forward.
  EXPECT_EQ(0.0, output->raw_value_);
  EXPECT_EQ(0.5, output->transformed_value_);
  EXPECT_EQ(0.0, hidden->raw_value_);
  EXPECT_EQ(0.5, hidden->transformed_value_);

  // Now run backprop and check the results.
  network.Backprop(in, target, learning_rate);
  // Previously, all inputs should be 0, so the delta at the output is 1.0 - 0.5 = 0.5
  // Both weights that are inputs to output should be adjusted by 0.5 * learning_rate *
  // SigmoidPrime(0).
  double delta1 = 0.5 * NeuralNetwork::SigmoidPrime(0);
  EXPECT_DOUBLE_EQ(delta1 * learning_rate, output->fixed_weight_->value);
  EXPECT_DOUBLE_EQ(delta1 * learning_rate * 0.5, output->weights_[0]->value);

  // The delta at the hidden layer.  The input to the hidden layer is 1.0.  The delta at the hidden
  // layer is the sum of forward weights multiplied by the delta of the neighbors.  The forward
  // weight is 0, so the hidden delta is 0.
  EXPECT_DOUBLE_EQ(0.0, hidden->fixed_weight_->value);
  EXPECT_DOUBLE_EQ(0.0, hidden->weights_[0]->value);

  // Now run one more iteration.
  network.FeedForward(in);
  EXPECT_EQ(0.0, hidden->raw_value_);
  EXPECT_EQ(0.5, hidden->transformed_value_);
  EXPECT_EQ(delta1 * learning_rate * 1.25, network.outputs_[0]->raw_value_);
  EXPECT_EQ(NeuralNetwork::Sigmoid(delta1 * learning_rate * 1.25),
            output->transformed_value_);
  double output_weight = output->weights_[0]->value;
  double output_fixed_weight = output->fixed_weight_->value;

  // Run backprop again and check the results.
  network.Backprop(in, target, learning_rate);
  // Now the error is 1 - network.outputs_[0]->transformed_value.
  double error = 1 - output->transformed_value_;
  double delta2 = error * NeuralNetwork::SigmoidPrime(output->raw_value_);
  EXPECT_EQ(output_fixed_weight + delta2 * learning_rate,
            output->fixed_weight_->value);
  EXPECT_EQ(output_weight + delta2 * learning_rate * 0.5,
            output->weights_[0]->value);
  double hidden_delta_2 = NeuralNetwork::SigmoidPrime(0.0) * output_weight * delta2;
  EXPECT_EQ(hidden_delta_2 * 1.0 * learning_rate, hidden->weights_[0]->value);
  EXPECT_EQ(hidden_delta_2 * 1.0 * learning_rate, hidden->fixed_weight_->value);
}

// Assume that network is a network with a single input and output.
// Generates examples using the formula y = slope * x + intercept.
// Checks that after training, the network's weights are within tolerance of
// the actual weights.
void TestLinear(double slope, double intercept, double tolerance, NeuralNetwork* network) {
  printf("TestLinear, slope: %f, intercept: %f\n", slope, intercept);
  // Create some fake inputs
  vector<Input> inputs;
  vector<Target> targets;
  for (int i = -10; i < 10; ++i) {
    Input input;
    input.values.push_back(i);
    inputs.push_back(input);
    Target target;
    target.values.push_back(NeuralNetwork::Sigmoid(slope * i + intercept));
    targets.push_back(target);
  }
  // Set all weights to 0
  for (size_t i = 0; i < network->weights_.size(); ++i) {
    network->weights_[i]->value = 0;
  }
  network->Train(inputs, targets, 1.0, 1000);
  double est_slope = network->outputs_[0]->weights_[0]->value;
  double est_intercept = network->outputs_[0]->fixed_weight_->value;
  printf("TestLinearResult, slope: %f, intercept: %f\n", est_slope, est_intercept);
  EXPECT_LT(fabs(slope - est_slope), tolerance);
  EXPECT_LT(fabs(intercept - est_intercept), tolerance);
}

// Test that runs the neural network for a single input and output node.  TestLinear checks that
// after a certain number of runs, the weights converge to the actual weights.
TEST(NeuralNetTest, SimpleTrain) {
  // Train a simple linear function with one input
  NeuralNetwork network;
  Node* input = new Node();
  network.AddNode(input, NeuralNetwork::INPUT);
  Node* output = new Node();
  output->AddInput(input, NULL, &network);
  network.AddNode(output, NeuralNetwork::OUTPUT);
  TestLinear(2, 1, 0.05, &network);
  TestLinear(2, 3, 0.05, &network);
  TestLinear(2, 5, 0.05, &network);
  TestLinear(2, 7, 0.15, &network);
}

TEST(NeuralNetTest, FeedForward) {
  // A simple single layer network.
  NeuralNetwork network;
  Node* input1 = new Node();
  Node* input2 = new Node();
  Node* output1 = new Node();
  network.AddNode(input1, NeuralNetwork::INPUT);
  network.AddNode(input2, NeuralNetwork::INPUT);
  output1->AddInput(input1, NULL, &network);
  output1->AddInput(input2, NULL, &network);
  output1->weights_[0]->value = 0.5;
  output1->weights_[1]->value = 1.5;
  output1->fixed_weight_->value = -3.0;
  Input input;
  input.values.push_back(1.0);
  input.values.push_back(2.0);
  network.AddNode(output1, NeuralNetwork::OUTPUT);
  network.FeedForward(input);
  double expected_total = 0.5 * 1.0 + 1.5 * 2.0 - 3.0;
  EXPECT_DOUBLE_EQ(expected_total, output1->raw_value_);
  EXPECT_DOUBLE_EQ(NeuralNetwork::Sigmoid(expected_total),
                   output1->transformed_value_);
}

TEST(NeuralNetTest, BasicAdd) {
  NeuralNetwork network;
  Node* node = new Node();
  network.AddNode(node, NeuralNetwork::INPUT);
  Node* node2 = new Node();
  Node* node3 = new Node();
  node2->AddInput(node3, NULL, &network);
  ASSERT_EQ((size_t) 1, node3->forward_neighbors_.size());
  ASSERT_EQ((size_t) 1, node3->forward_weights_.size());
  EXPECT_TRUE(node3->forward_weights_[0] == node2->weights_[0]);
  EXPECT_DEATH(network.AddNode(node2, NeuralNetwork::HIDDEN), ".*");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
