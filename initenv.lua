mfc = {}

require 'torch'
require 'nn'
require 'nngraph'
require 'image'
local opt = require 'opt'
have_qt = pcall(require, 'qt')
require 'mfc'
require "trace"
require "qtable"
require "kdtree/kdtree"

function setup ()
	torch.setdefaulttensortype('torch.FloatTensor')

	if opt.gpu and opt.gpu >= 0 then
		require 'cutorch'
        require 'cunn'

		if opt.gpu == 0 then
            local gpu_id = tonumber(os.getenv('GPU_ID'))
            if gpu_id then opt.gpu = gpu_id+1 end
        end
        if opt.gpu > 0 then cutorch.setDevice(opt.gpu) end
        opt.gpu = cutorch.getDevice()
        print('Using GPU device id:', opt.gpu-1)
	else
		opt.gpu = -1
        print('Using CPU code only. GPU device id:', opt.gpu)
	end

	local framework = require(opt.framework)
	local game_env = framework.GameEnvironment(opt)
	--local game_env_eval = framework.GameEnvironment(opt)
	local game_actions = game_env:getActions()
	opt.game_actions = game_actions
	local agent = mfc.ModelFreeEpisodic(opt)

	return game_env, game_actions, agent, opt
end

function print_in_line (str)
	io.write(str)
	io.flush()
end
