define(
  ["exports"],
  function(__exports__) {
    "use strict";
    function Visitor() {}

    Visitor.prototype = {
      constructor: Visitor,

      accept: function(object) {
        return this[object.type](object);
      }
    };

    __exports__["default"] = Visitor;
  });