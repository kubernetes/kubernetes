process.on('uncaughtException', function ( err ) {
  console.log('blah');
  throw err;
});

throw "on purpose"
