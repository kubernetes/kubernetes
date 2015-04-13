
// binaryType is truthy if there is support.. returns "blob" in new-ish chrome.
// plus.google.com/115535723976198353696/posts/ERN6zYozENV
// github.com/Modernizr/Modernizr/issues/370

Modernizr.addTest('websocketsbinary', function() {
  var protocol = 'https:'==location.protocol?'wss':'ws',
  protoBin;

  if('WebSocket' in window) {
    if( protoBin = 'binaryType' in WebSocket.prototype ) {
      return protoBin;
    }
    try {
      return !!(new WebSocket(protocol+'://.').binaryType);
    } catch (e){}
  }

  return false;
});
