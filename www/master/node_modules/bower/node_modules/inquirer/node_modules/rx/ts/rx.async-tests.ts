// Tests for RxJS-Async TypeScript definitions
// Tests by Igor Oleinikov <https://github.com/Igorbek>

/// <reference path="rx.async.d.ts" />

module Rx.Tests.Async {

	var obsNum: Rx.Observable<number>;
	var obsStr: Rx.Observable<string>;
	var sch: Rx.IScheduler;

	function start() {
		obsNum = Rx.Observable.start(()=> 10, obsStr, sch);
		obsNum = Rx.Observable.start(() => 10, obsStr);
		obsNum = Rx.Observable.start(()=> 10);
	}

	function toAsync() {
		obsNum = Rx.Observable.toAsync(()=> 1, sch)();
		obsNum = Rx.Observable.toAsync((a1: number)=> a1)(1);
        obsStr = <any>Rx.Observable.toAsync((a1: string, a2: number)=> a1 + a2.toFixed(0))("", 1);
		obsStr = <any>Rx.Observable.toAsync((a1: string, a2: number, a3: Date)=> a1 + a2.toFixed(0) + a3.toDateString())("", 1, new Date());
        obsStr = <any>Rx.Observable.toAsync((a1: string, a2: number, a3: Date, a4: boolean)=> a1 + a2.toFixed(0) + a3.toDateString() + (a4 ? 1 : 0))("", 1, new Date(), false);
	}

	function fromCallback() {
		// 0 arguments
		var func0: (cb: (result: number)=> void)=> void;
		obsNum = Rx.Observable.fromCallback(func0)();
		obsNum = Rx.Observable.fromCallback(func0, obsStr)();
		obsNum = Rx.Observable.fromCallback(func0, obsStr, (results: number[]) => results[0])();

		// 1 argument
		var func1: (a: string, cb: (result: number)=> void)=> number;
		obsNum = Rx.Observable.fromCallback(func1)("");
		obsNum = Rx.Observable.fromCallback(func1, {})("");
		obsNum = Rx.Observable.fromCallback(func1, {}, (results: number[]) => results[0])("");

		// 2 arguments
		var func2: (a: number, b: string, cb: (result: string) => number) => Date;
		obsStr = Rx.Observable.fromCallback(func2)(1, "");
		obsStr = Rx.Observable.fromCallback(func2, {})(1, "");
		obsStr = Rx.Observable.fromCallback(func2, {}, (results: string[]) => results[0])(1, "");

		// 3 arguments
		var func3: (a: number, b: string, c: boolean, cb: (result: string) => number) => Date;
		obsStr = Rx.Observable.fromCallback(func3)(1, "", true);
		obsStr = Rx.Observable.fromCallback(func3, {})(1, "", true);
		obsStr = Rx.Observable.fromCallback(func3, {}, (results: string[]) => results[0])(1, "", true);

		// multiple results
		var func0m: (cb: (result1: number, result2: number, result3: number) => void) => void;
		obsNum = Rx.Observable.fromCallback(func0m, obsStr, (results: number[]) => results[0])();
		var func1m: (a: string, cb: (result1: number, result2: number, result3: number) => void) => void;
		obsNum = Rx.Observable.fromCallback(func1m, obsStr, (results: number[]) => results[0])("");
		var func2m: (a: string, b: number, cb: (result1: string, result2: string, result3: string) => void) => void;
		obsStr = Rx.Observable.fromCallback(func2m, obsStr, (results: string[]) => results[0])("", 10);
	}

	function toPromise() {
		var promiseImpl: {
			new<T>(resolver: (resolvePromise: (value: T)=> void, rejectPromise: (reason: any)=> void)=> void): Rx.IPromise<T>;
		};

		Rx.config.Promise = promiseImpl;

		var p: IPromise<number> = obsNum.toPromise(promiseImpl);

		p = obsNum.toPromise();

		p = p.then(x=> x);
		p = p.then(x=> p);
		p = p.then(undefined, reason=> 10);
		p = p.then(undefined, reason=> p);

		var ps: IPromise<string> = p.then(undefined, reason=> "error");
		ps = p.then(x=> "");
		ps = p.then(x=> ps);
	}

	function startAsync() {
		var o: Rx.Observable<string> = Rx.Observable.startAsync(() => <Rx.IPromise<string>>null);
	}
}
