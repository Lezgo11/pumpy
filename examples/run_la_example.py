from pumpy.ode import simulate

if __name__ == "__main__":
    simulate(mode="la", beats=3, hr_bpm=70, log_dir="outputs/la_example", make_plots=True, E_la_min = 0.08)
