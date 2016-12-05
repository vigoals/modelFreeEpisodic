local opt = {
	framework="alewrap",
	env="qbert",
	game_path="/home/vigoals/atari_roms",
	actrep=4,
	random_starts=30,
	discount=0.99,
	eps_end=0.1,
	eps_endt=1000000,
	test_eps=0.05,
	steps=50000000,
	update_freq=1000,
	eval_freq=10000,
	eval_steps=10000,
	k=10
}

return opt
