function toggle(cls) {
	var els = document.getElementsByClassName(cls);
	var show = false
	for (var i = 0; i < els.length; i++) {
		if (els[i].className == 'test ' + cls) {
			if (els[i].style.display == 'none') {
				els[i].style.display = 'block';
				show = true;
			} else {
				els[i].style.display = 'none';
				show = false;
			}
		}
	}
	// UGLY HACK
	document.getElementsByClassName('total ' + cls)[0].style.color = show ? '#000000' : '#888888';
}

function defaultToggles() {
	toggle('okay');
	toggle('skipped');
}
window.onload = defaultToggles;
