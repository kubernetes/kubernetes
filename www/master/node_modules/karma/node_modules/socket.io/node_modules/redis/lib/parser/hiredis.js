/*global Buffer require exports console setTimeout */

var events = require("events"),
    util = require("../util"),
    hiredis = require("hiredis");

exports.debug_mode = false;
exports.name = "hiredis";

function HiredisReplyParser(options) {
    this.name = exports.name;
    this.options = options || {};
    this.reset();
    events.EventEmitter.call(this);
}

util.inherits(HiredisReplyParser, events.EventEmitter);

exports.Parser = HiredisReplyParser;

HiredisReplyParser.prototype.reset = function () {
    this.reader = new hiredis.Reader({
        return_buffers: this.options.return_buffers || false
    });
};

HiredisReplyParser.prototype.execute = function (data) {
    var reply;
    this.reader.feed(data);
    while (true) {
        try {
          reply = this.reader.get();
        } catch (err) {
          this.emit("error", err);
          break;
        }

        if (reply === undefined) break;

        if (reply && reply.constructor === Error) {
            this.emit("reply error", reply);
        } else {
            this.emit("reply", reply);
        }
    }
};
