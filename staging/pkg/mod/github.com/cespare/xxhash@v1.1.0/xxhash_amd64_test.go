// +build !appengine
// +build gc
// +build !purego

package xxhash

// TODO(caleb): Fix and re-enable with any ideas I get from
// https://groups.google.com/d/msg/golang-nuts/wb5I2tjrwoc/xCzk6uchBgAJ

//func TestSum64ASM(t *testing.T) {
//        for i := 0; i < 500; i++ {
//                b := make([]byte, i)
//                for j := range b {
//                        b[j] = byte(j)
//                }
//                pureGo := sum64Go(b)
//                asm := Sum64(b)
//                if pureGo != asm {
//                        t.Fatalf("[i=%d] pure go gave 0x%x; asm gave 0x%x", i, pureGo, asm)
//                }
//        }
//}

//func TestWriteBlocksASM(t *testing.T) {
//        x0 := New().(*xxh)
//        x1 := New().(*xxh)
//        for i := 32; i < 500; i++ {
//                b := make([]byte, i)
//                for j := range b {
//                        b[j] = byte(j)
//                }
//                pureGo := writeBlocksGo(x0, b)
//                asm := writeBlocks(x1, b)
//                if !reflect.DeepEqual(pureGo, asm) {
//                        t.Fatalf("[i=%d] pure go gave %v; asm gave %v", i, pureGo, asm)
//                }
//                if !reflect.DeepEqual(x0, x1) {
//                        t.Fatalf("[i=%d] pure go had state %v; asm had state %v", i, x0, x1)
//                }
//        }
//}
