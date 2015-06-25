package metrics

import "testing"

func BenchmarkEWMA(b *testing.B) {
	a := NewEWMA1()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Update(1)
		a.Tick()
	}
}

func TestEWMA1(t *testing.T) {
	a := NewEWMA1()
	a.Update(3)
	a.Tick()
	if rate := a.Rate(); 0.6 != rate {
		t.Errorf("initial a.Rate(): 0.6 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.22072766470286553 != rate {
		t.Errorf("1 minute a.Rate(): 0.22072766470286553 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.08120116994196772 != rate {
		t.Errorf("2 minute a.Rate(): 0.08120116994196772 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.029872241020718428 != rate {
		t.Errorf("3 minute a.Rate(): 0.029872241020718428 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.01098938333324054 != rate {
		t.Errorf("4 minute a.Rate(): 0.01098938333324054 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.004042768199451294 != rate {
		t.Errorf("5 minute a.Rate(): 0.004042768199451294 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.0014872513059998212 != rate {
		t.Errorf("6 minute a.Rate(): 0.0014872513059998212 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.0005471291793327122 != rate {
		t.Errorf("7 minute a.Rate(): 0.0005471291793327122 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.00020127757674150815 != rate {
		t.Errorf("8 minute a.Rate(): 0.00020127757674150815 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 7.404588245200814e-05 != rate {
		t.Errorf("9 minute a.Rate(): 7.404588245200814e-05 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 2.7239957857491083e-05 != rate {
		t.Errorf("10 minute a.Rate(): 2.7239957857491083e-05 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 1.0021020474147462e-05 != rate {
		t.Errorf("11 minute a.Rate(): 1.0021020474147462e-05 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 3.6865274119969525e-06 != rate {
		t.Errorf("12 minute a.Rate(): 3.6865274119969525e-06 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 1.3561976441886433e-06 != rate {
		t.Errorf("13 minute a.Rate(): 1.3561976441886433e-06 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 4.989172314621449e-07 != rate {
		t.Errorf("14 minute a.Rate(): 4.989172314621449e-07 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 1.8354139230109722e-07 != rate {
		t.Errorf("15 minute a.Rate(): 1.8354139230109722e-07 != %v\n", rate)
	}
}

func TestEWMA5(t *testing.T) {
	a := NewEWMA5()
	a.Update(3)
	a.Tick()
	if rate := a.Rate(); 0.6 != rate {
		t.Errorf("initial a.Rate(): 0.6 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.49123845184678905 != rate {
		t.Errorf("1 minute a.Rate(): 0.49123845184678905 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.4021920276213837 != rate {
		t.Errorf("2 minute a.Rate(): 0.4021920276213837 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.32928698165641596 != rate {
		t.Errorf("3 minute a.Rate(): 0.32928698165641596 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.269597378470333 != rate {
		t.Errorf("4 minute a.Rate(): 0.269597378470333 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.2207276647028654 != rate {
		t.Errorf("5 minute a.Rate(): 0.2207276647028654 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.18071652714732128 != rate {
		t.Errorf("6 minute a.Rate(): 0.18071652714732128 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.14795817836496392 != rate {
		t.Errorf("7 minute a.Rate(): 0.14795817836496392 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.12113791079679326 != rate {
		t.Errorf("8 minute a.Rate(): 0.12113791079679326 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.09917933293295193 != rate {
		t.Errorf("9 minute a.Rate(): 0.09917933293295193 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.08120116994196763 != rate {
		t.Errorf("10 minute a.Rate(): 0.08120116994196763 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.06648189501740036 != rate {
		t.Errorf("11 minute a.Rate(): 0.06648189501740036 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.05443077197364752 != rate {
		t.Errorf("12 minute a.Rate(): 0.05443077197364752 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.04456414692860035 != rate {
		t.Errorf("13 minute a.Rate(): 0.04456414692860035 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.03648603757513079 != rate {
		t.Errorf("14 minute a.Rate(): 0.03648603757513079 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.0298722410207183831020718428 != rate {
		t.Errorf("15 minute a.Rate(): 0.0298722410207183831020718428 != %v\n", rate)
	}
}

func TestEWMA15(t *testing.T) {
	a := NewEWMA15()
	a.Update(3)
	a.Tick()
	if rate := a.Rate(); 0.6 != rate {
		t.Errorf("initial a.Rate(): 0.6 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.5613041910189706 != rate {
		t.Errorf("1 minute a.Rate(): 0.5613041910189706 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.5251039914257684 != rate {
		t.Errorf("2 minute a.Rate(): 0.5251039914257684 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.4912384518467888184678905 != rate {
		t.Errorf("3 minute a.Rate(): 0.4912384518467888184678905 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.459557003018789 != rate {
		t.Errorf("4 minute a.Rate(): 0.459557003018789 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.4299187863442732 != rate {
		t.Errorf("5 minute a.Rate(): 0.4299187863442732 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.4021920276213831 != rate {
		t.Errorf("6 minute a.Rate(): 0.4021920276213831 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.37625345116383313 != rate {
		t.Errorf("7 minute a.Rate(): 0.37625345116383313 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.3519877317060185 != rate {
		t.Errorf("8 minute a.Rate(): 0.3519877317060185 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.3292869816564153165641596 != rate {
		t.Errorf("9 minute a.Rate(): 0.3292869816564153165641596 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.3080502714195546 != rate {
		t.Errorf("10 minute a.Rate(): 0.3080502714195546 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.2881831806538789 != rate {
		t.Errorf("11 minute a.Rate(): 0.2881831806538789 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.26959737847033216 != rate {
		t.Errorf("12 minute a.Rate(): 0.26959737847033216 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.2522102307052083 != rate {
		t.Errorf("13 minute a.Rate(): 0.2522102307052083 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.23594443252115815 != rate {
		t.Errorf("14 minute a.Rate(): 0.23594443252115815 != %v\n", rate)
	}
	elapseMinute(a)
	if rate := a.Rate(); 0.2207276647028646247028654470286553 != rate {
		t.Errorf("15 minute a.Rate(): 0.2207276647028646247028654470286553 != %v\n", rate)
	}
}

func elapseMinute(a EWMA) {
	for i := 0; i < 12; i++ {
		a.Tick()
	}
}
