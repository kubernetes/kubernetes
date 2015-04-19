'use strict';
var util = require('util');
var Transform = require('stream').Transform;

function ctor(options, transform) {
	util.inherits(FirstChunk, Transform);

	if (typeof options === 'function') {
		transform = options;
		options = {};
	}

	if (typeof transform !== 'function') {
		throw new Error('transform function required');
	}

	function FirstChunk(options2) {
		if (!(this instanceof FirstChunk)) {
			return new FirstChunk(options2);
		}

		Transform.call(this, options2);

		this._firstChunk = true;
		this._transformCalled = false;
		this._minSize = options.minSize;
	}

	FirstChunk.prototype._transform = function (chunk, enc, cb) {
		this._enc = enc;

		if (this._firstChunk) {
			this._firstChunk = false;

			if (this._minSize == null) {
				transform.call(this, chunk, enc, cb);
				this._transformCalled = true;
				return;
			}

			this._buffer = chunk;
			cb();
			return;
		}

		if (this._minSize == null) {
			this.push(chunk);
			cb();
			return;
		}

		if (this._buffer.length < this._minSize) {
			this._buffer = Buffer.concat([this._buffer, chunk]);
			cb();
			return;
		}

		if (this._buffer.length >= this._minSize) {
			transform.call(this, this._buffer.slice(), enc, function () {
				this.push(chunk);
				cb();
			}.bind(this));
			this._transformCalled = true;
			this._buffer = false;
			return;
		}

		this.push(chunk);
		cb();
	};

	FirstChunk.prototype._flush = function (cb) {
		if (!this._buffer) {
			cb();
			return;
		}

		if (this._transformCalled) {
			this.push(this._buffer);
			cb();
		} else {
			transform.call(this, this._buffer.slice(), this._enc, cb);
		}
	};

	return FirstChunk;
}

module.exports = function () {
	return ctor.apply(ctor, arguments)();
};

module.exports.ctor = ctor;
