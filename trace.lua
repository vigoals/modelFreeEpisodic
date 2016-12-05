if not mfc then
	require "initenv"
end

local trace = torch.class('Trace')

function trace:__init (size)
	self:clear()
	self.size = size
end

function trace:clear ()
	self.buf = {}
	self.tail = 0
	self.head = 0
end

function trace:add (state, action_index, reward, terminal)
	local tmp = {
		state=state,
		action_index=action_index,
		reward=reward,
		terminal=terminal
	}

	self.tail = self.tail + 1
	self.buf[self.tail] = tmp

	if self.tail - self.head > self.size then
		self.head = self.head + 1
		self.buf[self.head] = nil
	end
end
