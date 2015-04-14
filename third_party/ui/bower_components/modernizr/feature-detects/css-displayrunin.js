
// by alanhogan

// https://github.com/Modernizr/Modernizr/issues/198
// http://css-tricks.com/596-run-in/



Modernizr.testStyles(' #modernizr { display: run-in; } ', function(elem, rule){ 

  var ret = (window.getComputedStyle ?
         getComputedStyle(elem, null).getPropertyValue('display') :
         elem.currentStyle['display']);

  Modernizr.addTest('display-runin', ret == 'run-in');

});

