(function() {
  var CustomEvents, LiveReload, k;

  CustomEvents = require('./customevents');

  LiveReload = window.LiveReload = new (require('./livereload').LiveReload)(window);

  for (k in window) {
    if (k.match(/^LiveReloadPlugin/)) {
      LiveReload.addPlugin(window[k]);
    }
  }

  LiveReload.addPlugin(require('./less'));

  LiveReload.on('shutdown', function() {
    return delete window.LiveReload;
  });

  LiveReload.on('connect', function() {
    return CustomEvents.fire(document, 'LiveReloadConnect');
  });

  LiveReload.on('disconnect', function() {
    return CustomEvents.fire(document, 'LiveReloadDisconnect');
  });

  CustomEvents.bind(document, 'LiveReloadShutDown', function() {
    return LiveReload.shutDown();
  });

}).call(this);
