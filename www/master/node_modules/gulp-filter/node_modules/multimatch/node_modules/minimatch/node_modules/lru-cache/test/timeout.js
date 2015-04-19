var test = require("tap").test
var LRU = require("../")

var cache = LRU( {
    max: 1,
    maxAge: 500
} );

test('set the key', function (t) {
  cache.set( "1234", 1 );
  t.end()
})

for (var i = 0; i < 10; i ++) {
  test('get after ' + i + '00ms', function (t) {
    setTimeout(function () {
      t.equal(cache.get('1234'), 1)
      t.end()
    }, 100)
  })
}
