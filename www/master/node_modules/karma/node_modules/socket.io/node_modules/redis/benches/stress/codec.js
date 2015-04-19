var json = {
	encode: JSON.stringify,
	decode: JSON.parse
};

var MsgPack = require('node-msgpack');
msgpack = {
	encode: MsgPack.pack,
	decode: function(str) { return MsgPack.unpack(new Buffer(str)); }
};

bison = require('bison');

module.exports = json;
//module.exports = msgpack;
//module.exports = bison;
