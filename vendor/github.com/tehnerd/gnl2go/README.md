# gnl2go: Generic NetLink in Go

#### About:
This is go based lib to work with generic netlink socket's.
The lib was writen under heave influenc of FB's gnlpy
(so all kudos goes to FB's team  and all the blame to me)


in gnl2go.go you can find generic routines to work with gnetlink


in ipvs.go: lib to work with IPVS


in example/: few commands, which shows how to work with ipvs's lib

####TODOs:
bugfixes etc (i do know about incorrect ipv6struct to ipv6string conversion)
i dont use it (lib for ipvs) in production yet. not sure when i would. prob till that time i'd only fix
problem with ipv6 to string and any other minor bugs, which i'd bump into



####Output from example in example/:
```
sudo ipvsadm -C
sudo ipvsadm -ln
IP Virtual Server version 1.2.1 (size=4096)
Prot LocalAddress:Port Scheduler Flags
  -> RemoteAddress:Port           Forward Weight ActiveConn InActConn

sudo ./main
hi there
[]gnl2go.Pool(nil)
[]gnl2go.Pool{gnl2go.Pool{Service:gnl2go.Service{Proto:0x6, VIP:"2a020000:0:0:33:", Port:0x50, Sched:"wlc", FWMark:0x0, AF:0xa}, Dests:[]gnl2go.Dest(nil)}, gnl2go.Pool{Service:gnl2go.Service{Proto:0x6, VIP:"192.168.1.1", Port:0x50, Sched:"wrr", FWMark:0x0, AF:0x2}, Dests:[]gnl2go.Dest{gnl2go.Dest{IP:"127.0.0.11", Weight:10, Port:0x50, AF:0x2}}}, gnl2go.Pool{Service:gnl2go.Service{Proto:0x0, VIP:"", Port:0x0, Sched:"rr", FWMark:0x2, AF:0x0}, Dests:[]gnl2go.Dest(nil)}, gnl2go.Pool{Service:gnl2go.Service{Proto:0x0, VIP:"", Port:0x0, Sched:"wrr", FWMark:0x1, AF:0x0}, Dests:[]gnl2go.Dest(nil)}}
done

sudo ipvsadm -ln
IP Virtual Server version 1.2.1 (size=4096)
Prot LocalAddress:Port Scheduler Flags
  -> RemoteAddress:Port           Forward Weight ActiveConn InActConn
TCP  192.168.1.1:80 wrr
  -> 127.0.0.11:80                Tunnel  10     0          0         
  -> 127.0.0.13:80                Tunnel  20     0          0         
FWM  1 IPv6 wrr
  -> [fc00:1::12]:0               Tunnel  10     0          0         
  -> [fc00:2:3::12]:0             Tunnel  33     0          0         
```
