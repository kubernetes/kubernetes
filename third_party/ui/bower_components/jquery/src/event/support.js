define([
	"../var/support"
], function( support ) {

support.focusinBubbles = "onfocusin" in window;

return support;

});
