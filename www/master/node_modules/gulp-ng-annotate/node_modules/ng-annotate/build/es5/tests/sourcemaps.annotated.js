(function() {
  var ctrl4, x;

  x = "before";

  myMod.controller("ctrl1", ["ctrl1_param1", "ctrl1_param2", function(ctrl1_param1, ctrl1_param2) {
    return x = "ctrl1 body";
  }]);

  myMod.controller("ctrl2", ["ctrl2_param1", "ctrl2_param2", function(ctrl2_param1, ctrl2_param2) {
      return x = "ctrl2 body";
    }
  ]);

  myMod.controller("ctrl3", ["ctrl3_param1", "ctrl3_param2", function(ctrl3_param1, ctrl3_param2) {
      return x = "ctrl3 body";
    }
  ]);


  /* @ngInject */

  ctrl4 = function(ctrl4_param1, ctrl4_param2) {
    return x = "ctrl4 body";
  };
  ctrl4.$inject = ["ctrl4_param1", "ctrl4_param2"];

  x = "after";

}).call(this);
