// Stylable scrollbars detection
Modernizr.addTest('cssscrollbar', function() {

	var bool,

		styles = "#modernizr{overflow: scroll; width: 40px }#" +
			Modernizr._prefixes
				.join("scrollbar{width:0px}"+' #modernizr::')
				.split('#')
				.slice(1)
				.join('#') + "scrollbar{width:0px}";

	Modernizr.testStyles(styles, function(node) {
		bool = 'scrollWidth' in node && node.scrollWidth == 40;
	});

	return bool;

});
