// Type definitions for RxJS-Aggregates v2.2.28
// Project: http://rx.codeplex.com/
// Definitions by: Carl de Billy <http://carl.debilly.net/>, Igor Oleinikov <https://github.com/Igorbek>
// Definitions: https://github.com/borisyankov/DefinitelyTyped

///<reference path="rx.d.ts" />

declare module Rx {
	export interface Observable<T> {
		finalValue(): Observable<T>;
		aggregate(accumulator: (acc: T, value: T) => T): Observable<T>;
		aggregate<TAcc>(seed: TAcc, accumulator: (acc: TAcc, value: T) => TAcc): Observable<TAcc>;

		reduce(accumulator: (acc: T, value: T) => T): Observable<T>;
		reduce<TAcc>(accumulator: (acc: TAcc, value: T) => TAcc, seed: TAcc): Observable<TAcc>;		// TS0.9.5: won't work https://typescript.codeplex.com/discussions/471751

		any(predicate?: (value: T, index: number, source: Observable<T>) => boolean, thisArg?: any): Observable<boolean>;
		some(predicate?: (value: T, index: number, source: Observable<T>) => boolean, thisArg?: any): Observable<boolean>;	// alias for any

		isEmpty(): Observable<boolean>;
		all(predicate?: (value: T) => boolean, thisArg?: any): Observable<boolean>;
		every(predicate?: (value: T) => boolean, thisArg?: any): Observable<boolean>;	// alias for all
		contains(value: T): Observable<boolean>;
		contains<TOther>(value: TOther, comparer: (value1: T, value2: TOther) => boolean): Observable<boolean>;
		count(predicate?: (value: T, index: number, source: Observable<T>) => boolean, thisArg?: any): Observable<number>;
		sum(keySelector?: (value: T, index: number, source: Observable<T>) => number, thisArg?: any): Observable<number>;
		minBy<TKey>(keySelector: (item: T) => TKey, comparer: (value1: TKey, value2: TKey) => number): Observable<T>;
		minBy(keySelector: (item: T) => number): Observable<T>;
		min(comparer?: (value1: T, value2: T) => number): Observable<T>;
		maxBy<TKey>(keySelector: (item: T) => TKey, comparer: (value1: TKey, value2: TKey) => number): Observable<T>;
		maxBy(keySelector: (item: T) => number): Observable<T>;
		max(comparer?: (value1: T, value2: T) => number): Observable<number>;
		average(keySelector?: (value: T, index: number, source: Observable<T>) => number, thisArg?: any): Observable<number>;

		sequenceEqual<TOther>(second: Observable<TOther>, comparer: (value1: T, value2: TOther) => number): Observable<boolean>;
		sequenceEqual<TOther>(second: IPromise<TOther>, comparer: (value1: T, value2: TOther) => number): Observable<boolean>;
		sequenceEqual(second: Observable<T>): Observable<boolean>;
		sequenceEqual(second: IPromise<T>): Observable<boolean>;
		sequenceEqual<TOther>(second: TOther[], comparer: (value1: T, value2: TOther) => number): Observable<boolean>;
		sequenceEqual(second: T[]): Observable<boolean>;

		elementAt(index: number): Observable<T>;
		elementAtOrDefault(index: number, defaultValue?: T): Observable<T>;

		single(predicate?: (value: T, index: number, source: Observable<T>) => boolean, thisArg?: any): Observable<T>;
		singleOrDefault(predicate?: (value: T, index: number, source: Observable<T>) => boolean, defaultValue?: T, thisArg?: any): Observable<T>;

		first(predicate?: (value: T, index: number, source: Observable<T>) => boolean, thisArg?: any): Observable<T>;
		firstOrDefault(predicate?: (value: T, index: number, source: Observable<T>) => boolean, defaultValue?: T, thisArg?: any): Observable<T>;

		last(predicate?: (value: T, index: number, source: Observable<T>) => boolean, thisArg?: any): Observable<T>;
		lastOrDefault(predicate?: (value: T, index: number, source: Observable<T>) => boolean, defaultValue?: T, thisArg?: any): Observable<T>;

		find(predicate: (value: T, index: number, source: Observable<T>) => boolean, thisArg?: any): Observable<T>;
		findIndex(predicate: (value: T, index: number, source: Observable<T>) => boolean, thisArg?: any): Observable<number>;
	}
}

declare module "rx.aggregates" {
	export = Rx;
}
