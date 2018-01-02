package afero

import "testing"

func TestCopyOnWrite(t *testing.T) {
	var fs Fs
	var err error
	base := NewOsFs()
	roBase := NewReadOnlyFs(base)
	ufs := NewCopyOnWriteFs(roBase, NewMemMapFs())
	fs = ufs
	err = fs.MkdirAll("nonexistent/directory/", 0744)
	if err != nil {
		t.Error(err)
		return
	}
	_, err = fs.Create("nonexistent/directory/newfile")
	if err != nil {
		t.Error(err)
		return
	}

}
