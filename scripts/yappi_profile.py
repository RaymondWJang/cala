import yappi

from cala.main import main

try:
    yappi.set_clock_type("WALL")
    yappi.start()
    main(gui=True, spec="cala-unraveled")
    yappi.stop()
finally:
    stat = yappi.get_func_stats()
    ps = yappi.convert2pstats(stat)
    ps.dump_stats("prof/yappi3.prof")
