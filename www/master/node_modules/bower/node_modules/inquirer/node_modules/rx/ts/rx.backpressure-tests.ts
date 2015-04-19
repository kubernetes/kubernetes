// Tests for RxJS-BackPressure TypeScript definitions
// Tests by Igor Oleinikov <https://github.com/Igorbek>

///<reference path="rx.d.ts" />
///<reference path="rx.backpressure.d.ts" />

function testPausable() {
	var o: Rx.Observable<string>;

	var pauser = new Rx.Subject<boolean>();

	var p = o.pausable(pauser);
	p = o.pausableBuffered(pauser);
}

function testControlled() {
	var o: Rx.Observable<string>;
	var c = o.controlled();

	var d: Rx.IDisposable = c.request();
	d = c.request(5);
}
