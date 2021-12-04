//  _____                _           _   ______
// /  __ \              | |         | |  | ___ \
// | /  \/_ __ ___  __ _| |_ ___  __| |  | |_/ /_   _
// | |   | '__/ _ \/ _` | __/ _ \/ _` |  | ___ \ | | |
// | \__/\ | |  __/ (_| | ||  __/ (_| |  | |_/ / |_| |
//  \____/_|  \___|\__,_|\__\___|\__,_|  \____/ \__, |
//                                               __/ |
//                                              |___/
//   ___        _ _     ______     _       _
//  / _ \      (_) |    | ___ \   | |     | |
// / /_\ \_ __  _| | __ | |_/ /_ _| |_ ___| |
// |  _  | '_ \| | |/ / |  __/ _` | __/ _ \ |
// | | | | | | | |   <  | | | (_| | ||  __/ |
// \_| |_/_| |_|_|_|\_\ \_|  \__,_|\__\___|_|

// Constants
const IN = 0;
const HIDDEN = 1;
const OUT = 2;

// Available activation functions
const SIGMOID = 0;
const TANH = 1;

function sigmoid(x) {
  return 1 / (1 + exp(-x));
}

function randf(lo, hi) {
  return Math.random() * (hi + 1) + lo;
}

function randint(lo, hi) {
  return Math.floor(randf(lo, hi));
}

// Clones an object
function clone(target, map = new WeakMap()) {
  if (typeof target === "object") {
    let cloneTarget = Array.isArray(target) ? [] : {};
    if (map.get(target)) {
      return map.get(target);
    }
    map.set(target, cloneTarget);
    for (const key in target) {
      cloneTarget[key] = clone(target[key], map);
    }
    return cloneTarget;
  } else {
    return target;
  }
};

// Represents a connection between layers
class Connection {
  constructor(layerFrom, indexFrom, weight) {
    this.layerFrom = layerFrom;
    this.indexFrom = indexFrom;
    this.weight = weight;
  }

  copy() {
    let c = new Connection(this.layerFrom, this.indexFrom, this.weight);
    return c;
  }

}

// Represents a single neuron
class Neuron {
  constructor() {
    this.connections = [];
    this.outGoing = 0;
    this.outGoingConnections = [];
    this.value = 0;
    this.bias = 0;
  }

  addConnection(layerFrom, indexFrom, weight) {
    this.connections.push(new Connection(layerFrom, indexFrom, weight));
  }

  copy() {
    let c = new Neuron();
    for (let con of this.connections) {
      c.connections.push(con.copy());
    }
    return c;
  }
}

// Represents an entire Dynet network
class Dynet {
  constructor(inputs, outputs, hiddens=1, activation=SIGMOID) {
    // Must be greater then or equal to 1
    if (hiddens < 1) {
      hiddens = 1;
    }
    this.activation = activation;
    this.weightRange = 1;
    // Create the neurons
    this.inputs = [];
    this.outputs = [];
    this.hiddens = [];
    for (let i = 0; i < inputs; i ++) {
      this.inputs.push(new Neuron());
    }
    for (let i = 0; i < outputs; i ++) {
      this.outputs.push(new Neuron());
    }
    for (let i = 0; i < hiddens; i ++) {
      this.hiddens.push(new Neuron());
    }
  }

  activate(x) {
    // Run the activation function that was specified in the constructor
    if (this.activation == SIGMOID) {
      return sigmoid(x);
    }
    else if (this.activation == TANH) {
      return Math.tanh(x);
    }
  }

  addRandomInputToHiddenConnection() {
    let randomInputIndex = randint(0, this.inputs.length - 1);
    let randomHiddenIndex = randint(0, this.hiddens.length - 1);
    this.hiddens[randomHiddenIndex].addConnection(
      IN, randomInputIndex, randf(-this.weightRange, this.weightRange)
    );
    this.inputs[randomInputIndex].outGoing += 1;
  }

  addRandomInputToOutputConnection() {
    let randomInputIndex = randint(0, this.inputs.length - 1);
    let randomOutputIndex = randint(0, this.outputs.length - 1);

    this.outputs[randomOutputIndex].addConnection(
      IN, randomInputIndex, randf(-this.weightRange, this.weightRange)
    );
    this.inputs[randomInputIndex].outGoing += 1;
  }

  addRandomHiddenToOutputConnection() {
    let randomHiddenIndex = randint(0, this.hiddens.length - 1);
    let randomOutputIndex = randint(0, this.outputs.length - 1);
    this.outputs[randomOutputIndex].addConnection(
      HIDDEN, randomHiddenIndex, randf(-this.weightRange, this.weightRange)
    );
    this.hiddens[randomHiddenIndex].outGoing += 1;
  }

  addRandomHiddenToHiddenConnection() {
    let randomHiddenIndex = randint(0, this.hiddens.length - 1);
    let randomHidden2Index = randint(0, this.hiddens.length - 1);
    this.hiddens[randomHidden2Index].addConnection(
        HIDDEN, randomHiddenIndex, randf(-this.weightRange, this.weightRange)
    )
    this.hiddens[randomHiddenIndex].outGoing += 1;
  }

  addRandomConnection() {
    let r = randint(0, 4)
    if (r == 0){
      this.addRandomHiddenToHiddenConnection();
    }
    else if (r == 1){
      this.addRandomHiddenToOutputConnection();
    }
    else if (r == 2){
      this.addRandomInputToHiddenConnection();
    }
    else{
      this.addRandomInputToOutputConnection();
    }
  }

  removeRandomConnectionHiddens() {
    let randomHiddenIndex = randint(0, this.hiddens.length-1);
    if (this.hiddens[randomHiddenIndex].connections.length > 0) {
      let randomConnectionIndex = randint(0, this.hiddens[randomHiddenIndex].connections.length-1);
      this.hiddens[randomHiddenIndex].connections.pop(randomConnectionIndex);
    }
  }

  removeRandomConnectionOutputs() {
    let randomOutputIndex = randint(0, this.outputs.length-1);
    if (this.outputs[randomOutputIndex].connections.length > 0) {
      let randomConnectionIndex = randint(0, this.outputs[randomOutputIndex].connections.length-1);
      this.outputs[randomOutputIndex].connections.pop(randomConnectionIndex);
    }
  }

  removeRandomConnection() {
    // Remove a random connection
    if (random() < 0.5) {
      this.removeRandomConnectionHiddens();
    } else {
      this.removeRandomConnectionOutputs();
    }
  }

  constrainAllHiddenConnections(){
    for (let neuron of this.hiddens){
      for (let conn of neuron.connections){
        if (conn.weight < -this.weightRange){
          conn.weight = -this.weightRange
        }
        if (conn.weight > this.weightRange){
          conn.weight = this.weightRange
        }
      }
    }
  }

  constrainAllOutputConnections() {
    for (let neuron of this.outputs) {
      for (let conn of neuron.connections) {
        if (conn.weight < -this.weightRange) {
          conn.weight = -this.weightRange;
        }
        if (conn.weight > this.weightRange) {
          conn.weight = this.weightRange;
        }
      }
    }
  }

  mutateRandomConnection() {
    // Mutate a random connection
    if (random() < 0.5) {
      let randomHiddenIndex = randint(0, this.hiddens.length-1);
      if (this.hiddens[randomHiddenIndex].connections.length > 0) {
        let rConn = randint(0, this.hiddens[randomHiddenIndex].connections.length-1);
        this.hiddens[randomHiddenIndex].connections[rConn].weight += randf(-0.1, 0.1);

      }
    } else {
      let randomOutputIndex = randint(0, this.outputs.length-1);
      if (this.outputs[randomOutputIndex].connections.length > 0) {
        let rConn = randint(0, this.outputs[randomOutputIndex].connections.length-1);
        this.outputs[randomOutputIndex].connections[rConn].weight += randf(-0.1, 0.1);
      }
    }
  }

  weightedSumHiddens() {
    for (let neuron of this.hiddens) {
      neuron.value = 0;
      if (neuron.outGoing == 0) {
        //continue;
      }
      for (let conn of neuron.connections) {
        if (conn.layerFrom == IN) {
          neuron.value += conn.weight * this.inputs[conn.indexFrom].value;
        } else if (conn.layerFrom == HIDDEN) {
          neuron.value += conn.weight * this.hiddens[conn.indexFrom].value;
        }
      }
      neuron.value += neuron.bias;
      neuron.value = this.activate(neuron.value);
    }
  }

  weightedSumOutputs() {
    for (let neuron of this.outputs) {
        neuron.value = 0;
        for (let conn of neuron.connections) {
          if (conn.layerFrom == HIDDEN) {
            neuron.value += conn.weight * this.hiddens[conn.indexFrom].value;
          } else if (conn.layerFrom == IN) {
            neuron.value += conn.weight * this.inputs[conn.indexFrom].value;
          }
        }
        neuron.value = this.activate(neuron.value);
    }
  }
  fullyConnect(){
    for (let neuron in this.outputs){
      for (let n2 in this.hiddens){
        this.outputs[neuron].addConnection(HIDDEN, n2, randf(-this.weightRange, this.weightRange))
      }
    }
    for (let neuron in this.hiddens){
      for (let n2 in this.hiddens){
        this.hiddens[neuron].addConnection(HIDDEN, n2, randf(-this.weightRange, this.weightRange))
      }
    }
    for (let neuron in this.hiddens){
      for (let n2 in this.inputs){
        this.hiddens[neuron].addConnection(IN, n2, randf(-this.weightRange, this.weightRange))
      }
    }
  }
  mutateBias() {
    if (random() < 0.5) {
      let r = randint(0, this.outputs.length-1);
      this.outputs[r].bias += randf(-0.1, 0.1);
    } else if (this.hiddens.length > 0) {
      let r = randint(0, this.hiddens.length-1);
      this.hiddens[r].bias += randf(-0.1, 0.1);
    }
  }

  mutate(rate, repeats, modifyHiddens=true) {
    for (let i = 0; i < repeats; i ++) {
      if (random() < rate / 5) {
        this.addRandomConnection();
      }
      if (random() < rate) {
        this.mutateRandomConnection();
      }
      if (random() < rate / 20 && modifyHiddens) {
        this.hiddens.push(new Neuron());
      }
      if (random() < rate / 5) {
        this.removeRandomConnection();
      }
      if (random() < rate) {
        this.mutateBias();
      }
      this.constrainAllHiddenConnections()
      this.constrainAllOutputConnections()
    }
  }

  feedForward(ins) {
    let outputs = [];
    for (let i = 0; i < this.inputs.length; i++) {
      this.inputs[i].value = ins[i];
    }
    this.weightedSumHiddens();
    this.weightedSumOutputs();
    for (let i = 0; i < this.outputs.length; i++) {
      outputs.push(this.outputs[i].value);
    }
    return outputs;
  }

  printNetwork() {
    console.log("Network: ");
    let totalCons = 0;
    for (let i of this.inputs) {
      totalCons += i.connections.length;
    }
    for (let i of this.hiddens) {
      totalCons += i.connections.length;
    }
    for (let i of this.outputs) {
      totalCons += i.connections.length;
    }
    console.log("Connection count: " + totalCons);
  }

  copy() {
    let c = new Dynet(this.inputs.length, this.outputs.length, this.hiddens.length, this.activation);
    for (let i in this.inputs) {
      c.inputs[i] = this.inputs[i].copy();
    }
    for (let i in this.hiddens) {
      c.hiddens[i] = this.hiddens[i].copy();
    }
    for (let i in this.outputs) {
      c.outputs[i] = this.outputs[i].copy();
    }
    return c;
  }
}
