'use strict';

var memoize  = require('../..')
  , nextTick = require('next-tick');

module.exports = function () {
	return {
		Regular: {
			Sync: function (a) {
				var mfn, fn, i = 0;
				fn = function (x, y) {
					++i;
					return x + y;
				};
				mfn = memoize(fn, { max: 3 });

				a(mfn(3, 7), 10, "Result #1");
				a(i, 1, "Called #1");
				a(mfn(3, 7), 10, "Result #2");
				a(i, 1, "Called #2");
				a(mfn(5, 8), 13, "Result B #1");
				a(i, 2, "Called B #1");
				a(mfn(3, 7), 10, "Result #3");
				a(i, 2, "Called #3");
				a(mfn(5, 8), 13, "Result B #2");
				a(i, 2, "Called B #2");
				a(mfn(12, 4), 16, "Result C #1");
				a(i, 3, "Called C #1");
				a(mfn(3, 7), 10, "Result #4");
				a(i, 3, "Called #4");
				a(mfn(5, 8), 13, "Result B #3");
				a(i, 3, "Called B #3");

				a(mfn(77, 11), 88, "Result D #1"); // Delete 12, 4
				a(i, 4, "Called D #1");
				a(mfn(5, 8), 13, "Result B #4");
				a(i, 4, "Called B #4");
				a(mfn(12, 4), 16, "Result C #2"); // Delete 3, 7
				a(i, 5, "Called C #2");

				a(mfn(3, 7), 10, "Result #5"); // Delete 77, 11
				a(i, 6, "Called #5");
				a(mfn(77, 11), 88, "Result D #2"); // Delete 5, 8
				a(i, 7, "Called D #2");
				a(mfn(12, 4), 16, "Result C #3");
				a(i, 7, "Called C #3");

				a(mfn(5, 8), 13, "Result B #5"); // Delete 3, 7
				a(i, 8, "Called B #5");

				a(mfn(77, 11), 88, "Result D #3");
				a(i, 8, "Called D #3");

				mfn.delete(77, 11);
				a(mfn(77, 11), 88, "Result D #4");
				a(i, 9, "Called D #4");

				mfn.clear();
				a(mfn(5, 8), 13, "Result B #6");
				a(i, 10, "Called B #6");
				a(mfn(77, 11), 88, "Result D #5");
				a(i, 11, "Called D #5");
			},
			Async: function (a, d) {
				var mfn, fn, u = {}, i = 0;
				fn = function (x, y, cb) {
					nextTick(function () {
						++i;
						cb(null, x + y);
					});
					return u;
				};

				mfn = memoize(fn, { async: true, max: 3 });

				a(mfn(3, 7, function (err, res) {
					a.deep([err, res], [null, 10], "Result #1");
					a(i, 1, "Called #1");

					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #2");
						a(i, 1, "Called #2");

						a(mfn(5, 8, function (err, res) {
							a.deep([err, res], [null, 13], "Result B #1");
							a(i, 2, "Called B #1");

							a(mfn(3, 7, function (err, res) {
								a.deep([err, res], [null, 10], "Result #3");
								a(i, 2, "Called #3");

								a(mfn(5, 8, function (err, res) {
									a.deep([err, res], [null, 13], "Result B #2");
									a(i, 2, "Called B #2");

									a(mfn(12, 4, function (err, res) {
										a.deep([err, res], [null, 16], "Result C #1");
										a(i, 3, "Called C #1");

										a(mfn(3, 7, function (err, res) {
											a.deep([err, res], [null, 10], "Result #4");
											a(i, 3, "Called #4");

											a(mfn(5, 8, function (err, res) {
												a.deep([err, res], [null, 13], "Result B #3");
												a(i, 3, "Called B #3");

												a(mfn(77, 11, function (err, res) {
													a.deep([err, res], [null, 88], "Result D #1");
													a(i, 4, "Called D #1");

													a(mfn(5, 8, function (err, res) {
														a.deep([err, res], [null, 13], "Result B #4");
														a(i, 4, "Called B #4");

														a(mfn(12, 4, function (err, res) {
															a.deep([err, res], [null, 16], "Result C #2");
															a(i, 5, "Called C #2");

															a(mfn(3, 7, function (err, res) {
																a.deep([err, res], [null, 10], "Result #5");
																a(i, 6, "Called #5");

																a(mfn(77, 11, function (err, res) {
																	a.deep([err, res], [null, 88],
																		"Result D #2");
																	a(i, 7, "Called D #2");

																	a(mfn(12, 4, function (err, res) {
																		a.deep([err, res], [null, 16],
																			"Result C #3");
																		a(i, 7, "Called C #3");

																		a(mfn(5, 8, function (err, res) {
																			a.deep([err, res], [null, 13],
																				"Result B #5");
																			a(i, 8, "Called B #5");

																			a(mfn(77, 11, function (err, res) {
																				a.deep([err, res], [null, 88],
																					"Result D #3");
																				a(i, 8, "Called D #3");

																				mfn.delete(77, 11);
																				a(mfn(77, 11, function (err, res) {
																					a.deep([err, res], [null, 88],
																						"Result D #4");
																					a(i, 9, "Called D #4");

																					mfn.clear();
																					a(mfn(5, 8, function (err, res) {
																						a.deep([err, res], [null, 13],
																							"Result B #6");
																						a(i, 10, "Called B #6");

																						a(mfn(77, 11,
																							function (err, res) {
																								a.deep([err, res], [null, 88],
																									"Result D #5");
																								a(i, 11, "Called D #5");

																								d();
																							}), u, "Initial D #5");
																					}), u, "Initial B #6");
																				}), u, "Initial D #4");
																			}), u, "Initial D #3");
																		}), u, "Initial B #5");
																	}), u, "Initial C #3");
																}), u, "Initial D #2");
															}), u, "Initial #5");
														}), u, "Initial C #2");
													}), u, "Initial B #4");
												}), u, "Initial D #1");
											}), u, "Initial B #3");
										}), u, "Initial #4");
									}), u, "Initial C #1");
								}), u, "Initial B #2");
							}), u, "Initial #3");
						}), u, "Initial B #1");
					}), u, "Initial #2");
				}), u, "Initial #1");
			}
		},
		Primitive: {
			Sync: function (a) {
				var mfn, fn, i = 0;
				fn = function (x, y) {
					++i;
					return x + y;
				};
				mfn = memoize(fn, { primitive: true, max: 3 });

				a(mfn(3, 7), 10, "Result #1");
				a(i, 1, "Called #1");
				a(mfn(3, 7), 10, "Result #2");
				a(i, 1, "Called #2");
				a(mfn(5, 8), 13, "Result B #1");
				a(i, 2, "Called B #1");
				a(mfn(3, 7), 10, "Result #3");
				a(i, 2, "Called #3");
				a(mfn(5, 8), 13, "Result B #2");
				a(i, 2, "Called B #2");
				a(mfn(12, 4), 16, "Result C #1");
				a(i, 3, "Called C #1");
				a(mfn(3, 7), 10, "Result #4");
				a(i, 3, "Called #4");
				a(mfn(5, 8), 13, "Result B #3");
				a(i, 3, "Called B #3");

				a(mfn(77, 11), 88, "Result D #1"); // Delete 12, 4
				a(i, 4, "Called D #1");
				a(mfn(5, 8), 13, "Result B #4");
				a(i, 4, "Called B #4");
				a(mfn(12, 4), 16, "Result C #2"); // Delete 3, 7
				a(i, 5, "Called C #2");

				a(mfn(3, 7), 10, "Result #5"); // Delete 77, 11
				a(i, 6, "Called #5");
				a(mfn(77, 11), 88, "Result D #2"); // Delete 5, 8
				a(i, 7, "Called D #2");
				a(mfn(12, 4), 16, "Result C #3");
				a(i, 7, "Called C #3");

				a(mfn(5, 8), 13, "Result B #5"); // Delete 3, 7
				a(i, 8, "Called B #5");

				a(mfn(77, 11), 88, "Result D #3");
				a(i, 8, "Called D #3");

				mfn.delete(77, 11);
				a(mfn(77, 11), 88, "Result D #4");
				a(i, 9, "Called D #4");

				mfn.clear();
				a(mfn(5, 8), 13, "Result B #6");
				a(i, 10, "Called B #6");
				a(mfn(77, 11), 88, "Result D #5");
				a(i, 11, "Called D #5");
			},
			Async: function (a, d) {
				var mfn, fn, u = {}, i = 0;
				fn = function (x, y, cb) {
					nextTick(function () {
						++i;
						cb(null, x + y);
					});
					return u;
				};

				mfn = memoize(fn, { async: true, primitive: true, max: 3 });

				a(mfn(3, 7, function (err, res) {
					a.deep([err, res], [null, 10], "Result #1");
					a(i, 1, "Called #1");

					a(mfn(3, 7, function (err, res) {
						a.deep([err, res], [null, 10], "Result #2");
						a(i, 1, "Called #2");

						a(mfn(5, 8, function (err, res) {
							a.deep([err, res], [null, 13], "Result B #1");
							a(i, 2, "Called B #1");

							a(mfn(3, 7, function (err, res) {
								a.deep([err, res], [null, 10], "Result #3");
								a(i, 2, "Called #3");

								a(mfn(5, 8, function (err, res) {
									a.deep([err, res], [null, 13], "Result B #2");
									a(i, 2, "Called B #2");

									a(mfn(12, 4, function (err, res) {
										a.deep([err, res], [null, 16], "Result C #1");
										a(i, 3, "Called C #1");

										a(mfn(3, 7, function (err, res) {
											a.deep([err, res], [null, 10], "Result #4");
											a(i, 3, "Called #4");

											a(mfn(5, 8, function (err, res) {
												a.deep([err, res], [null, 13], "Result B #3");
												a(i, 3, "Called B #3");

												a(mfn(77, 11, function (err, res) {
													a.deep([err, res], [null, 88], "Result D #1");
													a(i, 4, "Called D #1");

													a(mfn(5, 8, function (err, res) {
														a.deep([err, res], [null, 13], "Result B #4");
														a(i, 4, "Called B #4");

														a(mfn(12, 4, function (err, res) {
															a.deep([err, res], [null, 16], "Result C #2");
															a(i, 5, "Called C #2");

															a(mfn(3, 7, function (err, res) {
																a.deep([err, res], [null, 10], "Result #5");
																a(i, 6, "Called #5");

																a(mfn(77, 11, function (err, res) {
																	a.deep([err, res], [null, 88],
																		"Result D #2");
																	a(i, 7, "Called D #2");

																	a(mfn(12, 4, function (err, res) {
																		a.deep([err, res], [null, 16],
																			"Result C #3");
																		a(i, 7, "Called C #3");

																		a(mfn(5, 8, function (err, res) {
																			a.deep([err, res], [null, 13],
																				"Result B #5");
																			a(i, 8, "Called B #5");

																			a(mfn(77, 11, function (err, res) {
																				a.deep([err, res], [null, 88],
																					"Result D #3");
																				a(i, 8, "Called D #3");

																				mfn.delete(77, 11);
																				a(mfn(77, 11, function (err, res) {
																					a.deep([err, res], [null, 88],
																						"Result D #4");
																					a(i, 9, "Called D #4");

																					mfn.clear();
																					a(mfn(5, 8, function (err, res) {
																						a.deep([err, res], [null, 13],
																							"Result B #6");
																						a(i, 10, "Called B #6");

																						a(mfn(77, 11,
																							function (err, res) {
																								a.deep([err, res], [null, 88],
																									"Result D #5");
																								a(i, 11, "Called D #5");

																								d();
																							}), u, "Initial D #5");
																					}), u, "Initial B #6");
																				}), u, "Initial D #4");
																			}), u, "Initial D #3");
																		}), u, "Initial B #5");
																	}), u, "Initial C #3");
																}), u, "Initial D #2");
															}), u, "Initial #5");
														}), u, "Initial C #2");
													}), u, "Initial B #4");
												}), u, "Initial D #1");
											}), u, "Initial B #3");
										}), u, "Initial #4");
									}), u, "Initial C #1");
								}), u, "Initial B #2");
							}), u, "Initial #3");
						}), u, "Initial B #1");
					}), u, "Initial #2");
				}), u, "Initial #1");
			}
		}
	};
};
