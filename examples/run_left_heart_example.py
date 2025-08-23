from pumpy.ode import simulate

if __name__ == "__main__":
    simulate(mode="la_lv", beats=3, hr_bpm=58, log_dir="outputs/left_heart_example", make_plots=False)
