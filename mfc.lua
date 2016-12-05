if not mfc then
	require 'initenv'
end

local mfc = torch.class('mfc.ModelFreeEpisodic')

function mfc:__init (args)
	self.game_actions = args.game_actions
	self.height = args.height or 84
	self.width = args.width or 84
	self.trace = Trace(args.update_freq + 10)
	self.qtable = QTable({k=args.k, n_actions=#self.game_actions})
	self.discount = args.discount or 0.99

	--for representation
	self.F = 100 or args.F	--转换后的维度
	self.A = torch.rand(self.F, self.height*self.width)

	self.last_state = None
	self.last_action_index = None
	self.last_train_step = 0
end

function mfc:pre_process (raw_state)
	local x = raw_state
	if x:dim() > 3 then
		x = x[1]
	end

	x = image.rgb2y(x)
	x = image.scale(x, self.width, self.height, 'bilinear')
	return x
end

function mfc:representation (state)
	local representation = state:clone():reshape(state:nElement(), 1)
	representation =
		torch.mm(self.A, representation):div(self.width*self.height)
	representation = representation:reshape(representation:nElement())
	return representation
end

function mfc:perceive (raw_state, reward, terminal, ep, eval, step)
	local eval = eval or true

	local state = self:pre_process(raw_state)
	local representation = self:representation(state)
	local action_index, q = self:policy(representation, ep)

	self.last_state = state
	self.last_action_index = action_index

	if reward > 1 then
		reward = 1
	elseif reward < -1 then
		reward = -1
	end

	self.trace:add(self.last_state, self.last_action_index, reward, terminal)

	return action_index, q
end

function mfc:q (representation)
	local q = self.qtable:q(representation)
	if not q then
		q = torch.zeros(self.F)
	else
		q = q:clone()
	end
	return q
end

function mfc:policy (representation, ep)
	if torch.uniform() < ep then
		return torch.random(#self.game_actions), 0
	else
		local q = self:q(representation)
		local max, max_index = q:max(1)
		return max_index[1], max[1]
	end
end

function mfc:train ()
	local last_q
	local q
	for i = self.trace.tail,
		math.max(self.last_train_step, self.trace.head) + 1, -1 do
		--print(i, self.trace.buf[i].reward)
		local tmp = self.trace.buf[i]
		local representation = self:representation(tmp.state)
		q = self:q(representation)

		local k = 0
		if last_q then
			k = tmp.reward
			if not tmp.terminal then
				k = k + self.discount*last_q
			end

			self.qtable:update(representation, tmp.action_index, k)
		end

		last_q = math.max(q:max(), k)
	end

	self.last_train_step = self.trace.tail
end

function mfc:report ()
	print("Number: " .. #self.qtable.buf)

	--self.qtable:report(20)
end
