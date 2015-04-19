var should = require('should');
var extend = require('../index');

describe('deep-extend', function() {

	it('can extend on 1 level', function() {
		var a = { hello: 1 };
		var b = { world: 2 };
		extend(a, b);
		a.should.eql({
			hello: 1,
			world: 2
		});
	});

	it('can extend on 2 levels', function() {
		var a = { person: { name: 'John' } };
		var b = { person: { age: 30 } };
		extend(a, b);
		a.should.eql({
			person: { name: 'John', age: 30 }
		});
	});

	it('can extend with Buffer values', function() {
		var a = { hello: 1 };
		var b = { value: new Buffer('world') };
		extend(a, b);
		a.should.eql({
			hello: 1,
			value: new Buffer('world')
		});
	});

	it('Buffer is cloned', function () {
		var a = { };
		var b = { value: new Buffer('foo') };
		extend(a, b);
		a.value.write('bar');
		a.value.toString().should.eql('bar');
		b.value.toString().should.eql('foo');
	});

	it('Date objects', function () {
		var a = { d: new Date() };
		var b = extend({}, a);
		b.d.should.instanceOf(Date);
	});

	it('Date object is cloned', function () {
		var a = { d: new Date() };
		var b = extend({}, a);
		b.d.setTime( (new Date()).getTime() + 100000 );
		b.d.getTime().should.not.eql( a.d.getTime() );
	});

});
