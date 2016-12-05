if not mfc then
	require "initenv"
end

local game_env, game_actions, agent, opt = setup(opt)

local step = 0
local screen, reward, terminal = game_env:getState()
local ep = 1
local win

print('Start at ' .. os.date() .. ':')
while step < opt.steps do
	step = step + 1

	print_in_line("step: " .. step .. "\r")
	win = have_qt and image.display({image=screen, win=win})

	if step < opt.eps_endt then
		ep = opt.eps_endt +
			(1 - opt.eps_endt)*(opt.eps_endt - step)/opt.eps_endt
	else
		ep = opt.eps_endt
	end

	local action_index =
		agent:perceive(screen, reward, terminal, ep, false, step)

	if not terminal then
		screen, reward, terminal =
			game_env:step(game_actions[action_index], true)
	else
		screen, reward, terminal = game_env:nextRandomGame()
	end

	if step%1000 == 0 then
		collectgarbage()
	end

	if step%opt.update_freq == 0 then
		agent:train()
	end

	if opt.eval_freq > 0 and step%opt.eval_freq == 0 then
		print("")
		print("eval in step " .. step)
		local eval_total_reward = 0
		local eval_episode = 0
		local eval_reward = 0

		screen, reward, terminal = game_env:newGame()
		for eval_step = 1, opt.eval_steps do
			local action_index, q =
				agent:perceive(screen, reward, terminal, opt.test_eps, true)
			eval_reward = eval_reward + reward
			win = have_qt and image.display({image=screen, win=win})

			print_in_line("eval step: " .. eval_step ..
				" q: " .. string.format("%.6f", q) .. "\r")

			if not terminal then
				screen, reward, terminal =
					game_env:step(game_actions[action_index])
			else
				screen, reward, terminal = game_env:newGame()
				eval_total_reward = eval_total_reward + eval_reward
				eval_episode = eval_episode + 1
				eval_reward = 0
			end
		end

		local eval_avg_reward = eval_total_reward/eval_episode
		print(string.format("total:%10.3f episode:%d avg:%10.3f",
			eval_total_reward, eval_episode, eval_avg_reward))

		print("Agetn info:")
		agent:report()

		reward = 0
		terminal = true
	end
end
