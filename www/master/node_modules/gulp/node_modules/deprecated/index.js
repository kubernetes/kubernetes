var deprecated = {
  method: function(msg, log, fn) {
    var called = false;
    return function(){
      if (!called) {
        called = true;
        log(msg);
      }
      return fn.apply(this, arguments);
    };
  },

  field: function(msg, log, parent, field, val) {
    var called = false;
    var getter = function(){
      if (!called) {
        called = true;
        log(msg);
      }
      return val;
    };
    var setter = function(v) {
      if (!called) {
        called = true;
        log(msg);
      }
      val = v;
      return v;
    };
    Object.defineProperty(parent, field, {
      get: getter,
      set: setter,
      enumerable: true
    });
    return;
  }
};

module.exports = deprecated;