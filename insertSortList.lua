if not mfc then
	require "initenv"
end

local isl = torch.class('InsertSortList')

function isl:__init (k)
	self.k = k
	self.buf = {}
end

function isl:insert (data)
	if #self.buf < self.k then
		self.buf[#self.buf + 1] = data
	else
		local i = self.k
		while i > 0 do
			if self.buf[i].dist > data.dist then
				i = i - 1
			else
				break
			end
		end

		i = i + 1
		if i <= self.k then
			table.insert(self.buf, i, data)
			table.remove(self.buf, self.k + 1)
		end
	end
end
