var to_array = require("./to_array");

// Queue class adapted from Tim Caswell's pattern library
// http://github.com/creationix/pattern/blob/master/lib/pattern/queue.js

function Queue() {
    this.tail = [];
    this.head = [];
    this.offset = 0;
}

Queue.prototype.shift = function () {
    if (this.offset === this.head.length) {
        var tmp = this.head;
        tmp.length = 0;
        this.head = this.tail;
        this.tail = tmp;
        this.offset = 0;
        if (this.head.length === 0) {
            return;
        }
    }
    return this.head[this.offset++]; // sorry, JSLint
};

Queue.prototype.push = function (item) {
    return this.tail.push(item);
};

Queue.prototype.forEach = function (fn, thisv) {
    var array = this.head.slice(this.offset), i, il;

    array.push.apply(array, this.tail);

    if (thisv) {
        for (i = 0, il = array.length; i < il; i += 1) {
            fn.call(thisv, array[i], i, array);
        }
    } else {
        for (i = 0, il = array.length; i < il; i += 1) {
            fn(array[i], i, array);
        }
    }

    return array;
};

Queue.prototype.getLength = function () {
    return this.head.length - this.offset + this.tail.length;
};
    
Object.defineProperty(Queue.prototype, 'length', {
    get: function () {
        return this.getLength();
    }
});


if(typeof module !== 'undefined' && module.exports) {
  module.exports = Queue;
}
