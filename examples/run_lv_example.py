from pumpy.ode import simulate

if __name__ == "__main__":
    simulate(mode="lv", beats=2, hr_bpm=70, log_dir="outputs/lv_example", make_plots=True)
