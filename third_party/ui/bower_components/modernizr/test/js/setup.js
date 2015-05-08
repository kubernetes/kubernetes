
// Avoid `console` errors in browsers that lack a console
if (!(window.console && console.log)) {
    (function() {
        var noop = function() {};
        var methods = ['assert', 'clear', 'count', 'debug', 'dir', 'dirxml', 'error', 'exception', 'group', 'groupCollapsed', 'groupEnd', 'info', 'log', 'markTimeline', 'profile', 'profileEnd', 'markTimeline', 'table', 'time', 'timeEnd', 'timeStamp', 'trace', 'warn'];
        var length = methods.length;
        var console = window.console = {};
        while (length--) {
            console[methods[length]] = noop;
        }
    }());
}

// test helper object
window.TEST = {
  // note some unique members of the Modernizr object
  inputs    : ['input','inputtypes', 'textarea'],
  audvid    : ['video','audio'],
  API       : ['addTest', 'mq', 'hasEvent', 'testProp', 'testAllProps', 'testStyles', '_prefixes', '_domPrefixes', '_cssomPrefixes', 'prefixed'],
  extraclass: ['js'],
  privates  : ['_enableHTML5','_version','_fontfaceready'],
  deprecated : [
                { oldish : 'crosswindowmessaging', newish : 'postmessage'},
                { oldish : 'historymanagement', newish : 'history'},
              ],

  // utility methods
  inArray: function(elem, array) {
      if (array.indexOf) {
          return array.indexOf(elem);
      }
      for (var i = 0, length = array.length; i < length; i++) {
          if (array[i] === elem) {
              return i;
          }
      }
      return -1;
  },
  trim : function(str){
    return str.replace(/^\s*/, "").replace(/\s*$/, "");
  }
};

