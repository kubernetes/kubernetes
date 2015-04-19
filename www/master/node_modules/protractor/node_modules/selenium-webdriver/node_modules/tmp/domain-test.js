var domain = require('domain');

//throw new Error('bazz');

var d = domain.create();
d.on('error', function ( e ) {
  console.log('error!!!', e);
});

d.run(function () {
  console.log('hey');
  throw new Error('bazz');
});
