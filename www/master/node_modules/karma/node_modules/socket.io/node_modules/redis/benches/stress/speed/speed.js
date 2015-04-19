var msgpack = require('node-msgpack');
var bison = require('bison');
var codec = {
	JSON: {
		encode: JSON.stringify,
		decode: JSON.parse
	},
	msgpack: {
		encode: msgpack.pack,
		decode: msgpack.unpack
	},
	bison: bison
};

var obj, l;

var s = '0';
for (var i = 0; i < 12; ++i) s += s;

obj = {
	foo: s,
	arrrrrr: [{a:1,b:false,c:null,d:1.0}, 1111, 2222, 33333333],
	rand: [],
	a: s,
	ccc: s,
	b: s + s + s
};
for (i = 0; i < 100; ++i) obj.rand.push(Math.random());
forObj(obj);

obj = {
	foo: s,
	arrrrrr: [{a:1,b:false,c:null,d:1.0}, 1111, 2222, 33333333],
	rand: []
};
for (i = 0; i < 100; ++i) obj.rand.push(Math.random());
forObj(obj);

obj = {
	foo: s,
	arrrrrr: [{a:1,b:false,c:null,d:1.0}, 1111, 2222, 33333333],
	rand: []
};
forObj(obj);

obj = {
	arrrrrr: [{a:1,b:false,c:null,d:1.0}, 1111, 2222, 33333333],
	rand: []
};
forObj(obj);

function run(obj, codec) {
	var t1 = Date.now();
	var n = 10000;
	for (var i = 0; i < n; ++i) {
		codec.decode(l = codec.encode(obj));
	}
	var t2 = Date.now();
	//console.log('DONE', n*1000/(t2-t1), 'codecs/sec, length=', l.length);
	return [n*1000/(t2-t1), l.length];
}

function series(obj, cname, n) {
	var rate = 0;
	var len = 0;
	for (var i = 0; i < n; ++i) {
		var r = run(obj, codec[cname]);
		rate += r[0];
		len += r[1];
	}
	rate /= n;
	len /= n;
	console.log(cname + '	' + rate + '	' + len);
	return [rate, len];
}

function forObj(obj) {
	var r = {
		JSON: series(obj, 'JSON', 20),
		msgpack: series(obj, 'msgpack', 20),
		bison: series(obj, 'bison', 20)
	};
	return r;
}
