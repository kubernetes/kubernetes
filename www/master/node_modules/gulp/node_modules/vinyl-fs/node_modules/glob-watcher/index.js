var gaze = require('gaze');
var EventEmitter = require('events').EventEmitter;

module.exports = function(glob, opts, cb) {
  var out = new EventEmitter();

  if (typeof opts === 'function') {
    cb = opts;
    opts = {};
  }

  var watcher = gaze(glob, opts, function(err, rwatcher){
    if (err) out.emit('error', err);
    rwatcher.on('all', function(evt, path, old){
      var outEvt = {type: evt, path: path};
      if(old) outEvt.old = old;
      out.emit('change', outEvt);
      if(cb) cb(outEvt);
    });
  });

  watcher.on('end', out.emit.bind(out, 'end'));
  watcher.on('error', out.emit.bind(out, 'error'));
  watcher.on('ready', out.emit.bind(out, 'ready'));
  watcher.on('nomatch', out.emit.bind(out, 'nomatch'));

  out.end = function(){
    return watcher.close();
  };
  out.add = function(){
    return watcher.add.apply(watcher, arguments);
  };
  out.remove = function(){
    return watcher.remove();
  };
  out._watcher = watcher;

  return out;
};
