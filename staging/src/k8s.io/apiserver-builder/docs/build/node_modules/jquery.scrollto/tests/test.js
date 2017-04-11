$.fn.test = function(){
	// Hide title on iframes
	if (window.self !== window.top) {
		$('h1').hide();
	}

	if (location.search === '?notest') {
		return this;
	}

	$.scrollTo.defaults.axis = 'xy';
	
	var root = this.is('iframe') ? this.contents() : $('body');
	root.find('#ua').html(
		navigator.userAgent +
		'<br />' +
		'document.compatMode is "' + document.compatMode + '"'
	);

	return this.scrollTo('max', 1000).scrollTo(0, 1000);
};
