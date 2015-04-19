/*!
 * ws: a node.js websocket client
 * Copyright(c) 2011 Einar Otto Stangvik <einaros@gmail.com>
 * MIT Licensed
 */

var util = require('util')
  , Validation = require('./Validation').Validation
  , ErrorCodes = require('./ErrorCodes')
  , BufferPool = require('./BufferPool')
  , bufferUtil = require('./BufferUtil').BufferUtil;

/**
 * HyBi Receiver implementation
 */

function Receiver () {
  // memory pool for fragmented messages
  var fragmentedPoolPrevUsed = -1;
  this.fragmentedBufferPool = new BufferPool(1024, function(db, length) {
    return db.used + length;
  }, function(db) {
    return fragmentedPoolPrevUsed = fragmentedPoolPrevUsed >= 0 ?
      (fragmentedPoolPrevUsed + db.used) / 2 :
      db.used;
  });

  // memory pool for unfragmented messages
  var unfragmentedPoolPrevUsed = -1;
  this.unfragmentedBufferPool = new BufferPool(1024, function(db, length) {
    return db.used + length;
  }, function(db) {
    return unfragmentedPoolPrevUsed = unfragmentedPoolPrevUsed >= 0 ?
      (unfragmentedPoolPrevUsed + db.used) / 2 :
      db.used;
  });

  this.state = {
    activeFragmentedOperation: null,
    lastFragment: false,
    masked: false,
    opcode: 0,
    fragmentedOperation: false
  };
  this.overflow = [];
  this.headerBuffer = new Buffer(10);
  this.expectOffset = 0;
  this.expectBuffer = null;
  this.expectHandler = null;
  this.currentMessage = [];
  this.expectHeader(2, this.processPacket);
  this.dead = false;

  this.onerror = function() {};
  this.ontext = function() {};
  this.onbinary = function() {};
  this.onclose = function() {};
  this.onping = function() {};
  this.onpong = function() {};
}

module.exports = Receiver;

/**
 * Add new data to the parser.
 *
 * @api public
 */

Receiver.prototype.add = function(data) {
  var dataLength = data.length;
  if (dataLength == 0) return;
  if (this.expectBuffer == null) {
    this.overflow.push(data);
    return;
  }
  var toRead = Math.min(dataLength, this.expectBuffer.length - this.expectOffset);
  fastCopy(toRead, data, this.expectBuffer, this.expectOffset);
  this.expectOffset += toRead;
  if (toRead < dataLength) {
    this.overflow.push(data.slice(toRead));
  }
  while (this.expectBuffer && this.expectOffset == this.expectBuffer.length) {
    var bufferForHandler = this.expectBuffer;
    this.expectBuffer = null;
    this.expectOffset = 0;
    this.expectHandler.call(this, bufferForHandler);
  }
};

/**
 * Releases all resources used by the receiver.
 *
 * @api public
 */

Receiver.prototype.cleanup = function() {
  this.dead = true;
  this.overflow = null;
  this.headerBuffer = null;
  this.expectBuffer = null;
  this.expectHandler = null;
  this.unfragmentedBufferPool = null;
  this.fragmentedBufferPool = null;
  this.state = null;
  this.currentMessage = null;
  this.onerror = null;
  this.ontext = null;
  this.onbinary = null;
  this.onclose = null;
  this.onping = null;
  this.onpong = null;
};

/**
 * Waits for a certain amount of header bytes to be available, then fires a callback.
 *
 * @api private
 */

Receiver.prototype.expectHeader = function(length, handler) {
  if (length == 0) {
    handler(null);
    return;
  }
  this.expectBuffer = this.headerBuffer.slice(this.expectOffset, this.expectOffset + length);
  this.expectHandler = handler;
  var toRead = length;
  while (toRead > 0 && this.overflow.length > 0) {
    var fromOverflow = this.overflow.pop();
    if (toRead < fromOverflow.length) this.overflow.push(fromOverflow.slice(toRead));
    var read = Math.min(fromOverflow.length, toRead);
    fastCopy(read, fromOverflow, this.expectBuffer, this.expectOffset);
    this.expectOffset += read;
    toRead -= read;
  }
};

/**
 * Waits for a certain amount of data bytes to be available, then fires a callback.
 *
 * @api private
 */

Receiver.prototype.expectData = function(length, handler) {
  if (length == 0) {
    handler(null);
    return;
  }
  this.expectBuffer = this.allocateFromPool(length, this.state.fragmentedOperation);
  this.expectHandler = handler;
  var toRead = length;
  while (toRead > 0 && this.overflow.length > 0) {
    var fromOverflow = this.overflow.pop();
    if (toRead < fromOverflow.length) this.overflow.push(fromOverflow.slice(toRead));
    var read = Math.min(fromOverflow.length, toRead);
    fastCopy(read, fromOverflow, this.expectBuffer, this.expectOffset);
    this.expectOffset += read;
    toRead -= read;
  }
};

/**
 * Allocates memory from the buffer pool.
 *
 * @api private
 */

Receiver.prototype.allocateFromPool = function(length, isFragmented) {
  return (isFragmented ? this.fragmentedBufferPool : this.unfragmentedBufferPool).get(length);
};

/**
 * Start processing a new packet.
 *
 * @api private
 */

Receiver.prototype.processPacket = function (data) {
  if ((data[0] & 0x70) != 0) {
    this.error('reserved fields must be empty', 1002);
    return;
  }
  this.state.lastFragment = (data[0] & 0x80) == 0x80;
  this.state.masked = (data[1] & 0x80) == 0x80;
  var opcode = data[0] & 0xf;
  if (opcode === 0) {
    // continuation frame
    this.state.fragmentedOperation = true;
    this.state.opcode = this.state.activeFragmentedOperation;
    if (!(this.state.opcode == 1 || this.state.opcode == 2)) {
      this.error('continuation frame cannot follow current opcode', 1002);
      return;
    }
  }
  else {
    if (opcode < 3 && this.state.activeFragmentedOperation != null) {
      this.error('data frames after the initial data frame must have opcode 0', 1002);
      return;
    }
    this.state.opcode = opcode;
    if (this.state.lastFragment === false) {
      this.state.fragmentedOperation = true;
      this.state.activeFragmentedOperation = opcode;
    }
    else this.state.fragmentedOperation = false;
  }
  var handler = opcodes[this.state.opcode];
  if (typeof handler == 'undefined') this.error('no handler for opcode ' + this.state.opcode, 1002);
  else {
    handler.start.call(this, data);
  }
};

/**
 * Endprocessing a packet.
 *
 * @api private
 */

Receiver.prototype.endPacket = function() {
  if (!this.state.fragmentedOperation) this.unfragmentedBufferPool.reset(true);
  else if (this.state.lastFragment) this.fragmentedBufferPool.reset(false);
  this.expectOffset = 0;
  this.expectBuffer = null;
  this.expectHandler = null;
  if (this.state.lastFragment && this.state.opcode === this.state.activeFragmentedOperation) {
    // end current fragmented operation
    this.state.activeFragmentedOperation = null;
  }
  this.state.lastFragment = false;
  this.state.opcode = this.state.activeFragmentedOperation != null ? this.state.activeFragmentedOperation : 0;
  this.state.masked = false;
  this.expectHeader(2, this.processPacket);
};

/**
 * Reset the parser state.
 *
 * @api private
 */

Receiver.prototype.reset = function() {
  if (this.dead) return;
  this.state = {
    activeFragmentedOperation: null,
    lastFragment: false,
    masked: false,
    opcode: 0,
    fragmentedOperation: false
  };
  this.fragmentedBufferPool.reset(true);
  this.unfragmentedBufferPool.reset(true);
  this.expectOffset = 0;
  this.expectBuffer = null;
  this.expectHandler = null;
  this.overflow = [];
  this.currentMessage = [];
};

/**
 * Unmask received data.
 *
 * @api private
 */

Receiver.prototype.unmask = function (mask, buf, binary) {
  if (mask != null && buf != null) bufferUtil.unmask(buf, mask);
  if (binary) return buf;
  return buf != null ? buf.toString('utf8') : '';
};

/**
 * Concatenates a list of buffers.
 *
 * @api private
 */

Receiver.prototype.concatBuffers = function(buffers) {
  var length = 0;
  for (var i = 0, l = buffers.length; i < l; ++i) length += buffers[i].length;
  var mergedBuffer = new Buffer(length);
  bufferUtil.merge(mergedBuffer, buffers);
  return mergedBuffer;
};

/**
 * Handles an error
 *
 * @api private
 */

Receiver.prototype.error = function (reason, protocolErrorCode) {
  this.reset();
  this.onerror(reason, protocolErrorCode);
  return this;
};

/**
 * Buffer utilities
 */

function readUInt16BE(start) {
  return (this[start]<<8) +
         this[start+1];
}

function readUInt32BE(start) {
  return (this[start]<<24) +
         (this[start+1]<<16) +
         (this[start+2]<<8) +
         this[start+3];
}

function fastCopy(length, srcBuffer, dstBuffer, dstOffset) {
  switch (length) {
    default: srcBuffer.copy(dstBuffer, dstOffset, 0, length); break;
    case 16: dstBuffer[dstOffset+15] = srcBuffer[15];
    case 15: dstBuffer[dstOffset+14] = srcBuffer[14];
    case 14: dstBuffer[dstOffset+13] = srcBuffer[13];
    case 13: dstBuffer[dstOffset+12] = srcBuffer[12];
    case 12: dstBuffer[dstOffset+11] = srcBuffer[11];
    case 11: dstBuffer[dstOffset+10] = srcBuffer[10];
    case 10: dstBuffer[dstOffset+9] = srcBuffer[9];
    case 9: dstBuffer[dstOffset+8] = srcBuffer[8];
    case 8: dstBuffer[dstOffset+7] = srcBuffer[7];
    case 7: dstBuffer[dstOffset+6] = srcBuffer[6];
    case 6: dstBuffer[dstOffset+5] = srcBuffer[5];
    case 5: dstBuffer[dstOffset+4] = srcBuffer[4];
    case 4: dstBuffer[dstOffset+3] = srcBuffer[3];
    case 3: dstBuffer[dstOffset+2] = srcBuffer[2];
    case 2: dstBuffer[dstOffset+1] = srcBuffer[1];
    case 1: dstBuffer[dstOffset] = srcBuffer[0];
  }
}

/**
 * Opcode handlers
 */

var opcodes = {
  // text
  '1': {
    start: function(data) {
      var self = this;
      // decode length
      var firstLength = data[1] & 0x7f;
      if (firstLength < 126) {
        opcodes['1'].getData.call(self, firstLength);
      }
      else if (firstLength == 126) {
        self.expectHeader(2, function(data) {
          opcodes['1'].getData.call(self, readUInt16BE.call(data, 0));
        });
      }
      else if (firstLength == 127) {
        self.expectHeader(8, function(data) {
          if (readUInt32BE.call(data, 0) != 0) {
            self.error('packets with length spanning more than 32 bit is currently not supported', 1008);
            return;
          }
          opcodes['1'].getData.call(self, readUInt32BE.call(data, 4));
        });
      }
    },
    getData: function(length) {
      var self = this;
      if (self.state.masked) {
        self.expectHeader(4, function(data) {
          var mask = data;
          self.expectData(length, function(data) {
            opcodes['1'].finish.call(self, mask, data);
          });
        });
      }
      else {
        self.expectData(length, function(data) {
          opcodes['1'].finish.call(self, null, data);
        });
      }
    },
    finish: function(mask, data) {
      var packet = this.unmask(mask, data, true);
      if (packet != null) this.currentMessage.push(packet);
      if (this.state.lastFragment) {
        var messageBuffer = this.concatBuffers(this.currentMessage);
        if (!Validation.isValidUTF8(messageBuffer)) {
          this.error('invalid utf8 sequence', 1007);
          return;
        }
        this.ontext(messageBuffer.toString('utf8'), {masked: this.state.masked, buffer: messageBuffer});
        this.currentMessage = [];
      }
      this.endPacket();
    }
  },
  // binary
  '2': {
    start: function(data) {
      var self = this;
      // decode length
      var firstLength = data[1] & 0x7f;
      if (firstLength < 126) {
        opcodes['2'].getData.call(self, firstLength);
      }
      else if (firstLength == 126) {
        self.expectHeader(2, function(data) {
          opcodes['2'].getData.call(self, readUInt16BE.call(data, 0));
        });
      }
      else if (firstLength == 127) {
        self.expectHeader(8, function(data) {
          if (readUInt32BE.call(data, 0) != 0) {
            self.error('packets with length spanning more than 32 bit is currently not supported', 1008);
            return;
          }
          opcodes['2'].getData.call(self, readUInt32BE.call(data, 4, true));
        });
      }
    },
    getData: function(length) {
      var self = this;
      if (self.state.masked) {
        self.expectHeader(4, function(data) {
          var mask = data;
          self.expectData(length, function(data) {
            opcodes['2'].finish.call(self, mask, data);
          });
        });
      }
      else {
        self.expectData(length, function(data) {
          opcodes['2'].finish.call(self, null, data);
        });
      }
    },
    finish: function(mask, data) {
      var packet = this.unmask(mask, data, true);
      if (packet != null) this.currentMessage.push(packet);
      if (this.state.lastFragment) {
        var messageBuffer = this.concatBuffers(this.currentMessage);
        this.onbinary(messageBuffer, {masked: this.state.masked, buffer: messageBuffer});
        this.currentMessage = [];
      }
      this.endPacket();
    }
  },
  // close
  '8': {
    start: function(data) {
      var self = this;
      if (self.state.lastFragment == false) {
        self.error('fragmented close is not supported', 1002);
        return;
      }

      // decode length
      var firstLength = data[1] & 0x7f;
      if (firstLength < 126) {
        opcodes['8'].getData.call(self, firstLength);
      }
      else {
        self.error('control frames cannot have more than 125 bytes of data', 1002);
      }
    },
    getData: function(length) {
      var self = this;
      if (self.state.masked) {
        self.expectHeader(4, function(data) {
          var mask = data;
          self.expectData(length, function(data) {
            opcodes['8'].finish.call(self, mask, data);
          });
        });
      }
      else {
        self.expectData(length, function(data) {
          opcodes['8'].finish.call(self, null, data);
        });
      }
    },
    finish: function(mask, data) {
      var self = this;
      data = self.unmask(mask, data, true);
      if (data && data.length == 1) {
        self.error('close packets with data must be at least two bytes long', 1002);
        return;
      }
      var code = data && data.length > 1 ? readUInt16BE.call(data, 0) : 1000;
      if (!ErrorCodes.isValidErrorCode(code)) {
        self.error('invalid error code', 1002);
        return;
      }
      var message = '';
      if (data && data.length > 2) {
        var messageBuffer = data.slice(2);
        if (!Validation.isValidUTF8(messageBuffer)) {
          self.error('invalid utf8 sequence', 1007);
          return;
        }
        message = messageBuffer.toString('utf8');
      }
      this.onclose(code, message, {masked: self.state.masked});
      this.reset();
    },
  },
  // ping
  '9': {
    start: function(data) {
      var self = this;
      if (self.state.lastFragment == false) {
        self.error('fragmented ping is not supported', 1002);
        return;
      }

      // decode length
      var firstLength = data[1] & 0x7f;
      if (firstLength < 126) {
        opcodes['9'].getData.call(self, firstLength);
      }
      else {
        self.error('control frames cannot have more than 125 bytes of data', 1002);
      }
    },
    getData: function(length) {
      var self = this;
      if (self.state.masked) {
        self.expectHeader(4, function(data) {
          var mask = data;
          self.expectData(length, function(data) {
            opcodes['9'].finish.call(self, mask, data);
          });
        });
      }
      else {
        self.expectData(length, function(data) {
          opcodes['9'].finish.call(self, null, data);
        });
      }
    },
    finish: function(mask, data) {
      this.onping(this.unmask(mask, data, true), {masked: this.state.masked, binary: true});
      this.endPacket();
    }
  },
  // pong
  '10': {
    start: function(data) {
      var self = this;
      if (self.state.lastFragment == false) {
        self.error('fragmented pong is not supported', 1002);
        return;
      }

      // decode length
      var firstLength = data[1] & 0x7f;
      if (firstLength < 126) {
        opcodes['10'].getData.call(self, firstLength);
      }
      else {
        self.error('control frames cannot have more than 125 bytes of data', 1002);
      }
    },
    getData: function(length) {
      var self = this;
      if (this.state.masked) {
        this.expectHeader(4, function(data) {
          var mask = data;
          self.expectData(length, function(data) {
            opcodes['10'].finish.call(self, mask, data);
          });
        });
      }
      else {
        this.expectData(length, function(data) {
          opcodes['10'].finish.call(self, null, data);
        });
      }
    },
    finish: function(mask, data) {
      this.onpong(this.unmask(mask, data, true), {masked: this.state.masked, binary: true});
      this.endPacket();
    }
  }
}
