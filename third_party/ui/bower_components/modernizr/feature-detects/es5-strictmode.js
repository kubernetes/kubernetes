// strict mode

// test by @kangax

Modernizr.addTest('strictmode', function(){
	return (function(){ "use strict"; return !this; })(); 
});