var cache = require('./index')
;

cache.debug(false);

cache.put('a', true);
console.log('true == '+cache.get('a'));
cache.clear();
console.log('null == '+cache.get('a'));

console.log('null == '+cache.get('a'));
console.log('0 == '+cache.size());

cache.put('a', 'b', 3000);
console.log('1 == '+cache.size());

console.log('b == '+cache.get('a'));

var complicated = ['a',{'b':'c','d':['e',3]},'@'];
cache.put(complicated, true);
console.log('true == '+cache.get(complicated));
cache.del(complicated);
console.log('null == '+cache.get(complicated));

console.log('1 == '+cache.size());
cache.put(0, 0);
console.log('2 == '+cache.size());
cache.del(0);

cache.put('c', 'd', 1000, function() { 
  console.log('callback was called');
});

setTimeout(function() {
  console.log('b == '+cache.get('a'));
}, 2000);

setTimeout(function() {
  console.log('null == '+cache.get('a'));
  console.log('0 == '+cache.size());
}, 4000);
  
setTimeout(function() {
	console.log('Cache hits: ' + cache.hits());
	console.log('Cache misses: ' + cache.misses());	
}, 5000);

cache.put('timeout', 'timeout', 2000);

setTimeout(function() {
	console.log('timeout == '+cache.get('timeout'));
	cache.put('timeout', 'timeout-re', 2000); // Cancel timeout on NEW put
}, 1000);

setTimeout(function() {
	console.log('timeout-re == '+cache.get('timeout'));
}, 3000);
