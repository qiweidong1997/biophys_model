% 2/27/17
% I am changing the model neurons into leaky integrate-and-fire 
% The drawback is no intrinsic differentiation between excitatory/inhibitory neurons
% 3/2/17
% Changing delay times to reflect slow excitation/fast inhibition (i.e. Lim
% and Goldman 2013)
% adding a separate delay variable for inhibitory neurons
% Also added background noise to inhibitory neurons
% 3/3/17
% Do I need spike frequency adaptation (For RS)? I shouldn't if neurons only fire
% for a small period of time. However, it would be good to add it to
% differentiate RS and FS neurons (got this idea from Wang 1999).
% Additionally, this might prevent multiple 'cycles' to be activated at one
% time
% Also in this paper, it says realistic elevated firing rates are 10-50Hz
% 3/9/17
% Attempting to show irregular firing in downstream neuron
% 3/13/17
% Adding multiple downstream neurons, trying to show that they will fire in
% a specific order when population is interrogated

clear;
% Using phasic spiking (PS) parameters
Ne=4000;  %Excitatory
Ni=1000;  %Inhibitory
Nq=20;   %Readout

E_L = -70; %resting membrane potential [mV]
V_th = -55; %spike threshold [mV]
V_reset = -75; %value to reset voltage to after a spike [mV]
V_spike = 20; %value to draw a spike to, when cell spikes [mV]
R_m = 10; %membrane resistance [MOhm]
tau = 10; %membrane time constant [ms]

ve = -65; %Excitatory
vi = -65; %Inhibitory
vq = -65; %Readout

% Columns presynaptic, rows postsynaptic
% initialize synapse values to zero, "S" is "Synapse"
See = 0.0*ones(Ne,Ne); % intialize to 0
Sei = 0.*ones(Ni,Ne); % initialize to 0
Sie = -0.0*ones(Ne,Ni); % intialize to 0
% Seq = repmat([0.001:0.001:0.02]',1,Ne); % synapses are nonuniform
Seq = 0.03*ones(Nq,Ne); % all synapses are uniform
Siq = -0.008*ones(Nq,Ni); %Q: What are the 0.008, weights? 

% initialize to random
eei = 0.01;                                                                                                 
See = eei*rand(Ne); % intialize to 0
Sei = eei*rand(Ni,Ne); % initialize to 0
Sie = -0.0*ones(Ne,Ni); % intialize to 0 %Q: Are these like arrays that represent, "from e (excitatory) to i (inhibitory)"?
% Seq = repmat([0.001:0.001:0.02]',1,Ne); % synapses are nonuniform

% Seq = 0.03*ones(Nq,Ne); % all synapses are uniform
% Siq = -0.008*ones(Nq,Ni);

numPools = 25; %Q: Do pool sizes mean anything or help with anything?
Nre = Ne/numPools; % neurons per pool in rows
Nce = Ne/numPools; % neurons per pool in columns
Nri = Ni/numPools;
Nci = Ni/numPools;

%% Define the relationships between neurons
% define synapses in ring structure

% excitatory - excitatory weight
ee = 0.12; % 0.125

% inhibitory - excitatory weight
ie = -0.205; % -0.2
for i = 1:numPools-1
    %Q: I'm confused about the three formulas here... What do they mean?
    %Where should I find the references for them? Do I need to change these
    %for my part (making the model being able to be trained with inputs)?
    % Synapses _ excitatory - excitatory
    See(1+Ne-((numPools-1)/numPools)*Ne+Nre*(i-1):Ne-((numPools-1)/numPools)*Ne+Nre*i,1+Nce*(i-1):Nce*i) = ee;
%     Sei(1+Ni-((numPools-1)/numPools)*Ni+Nri*(i-1):Ni-((numPools-1)/numPools)*Ni+Nri*i,1+Nce*(i-1):Nce*i) = 0.15;
    
    % Synapses _ excitatory - inhibitory
    Sei(1+Nri*(i-1):Nri*i,1+Nce*(i-1):Nce*i) = 0.13; % Recurrent
%     See(1+Nre*(i-1):Nre*i,1+Nce*(i-1):Nce*i) = 0.02; % Recurrent

    % Synapses _ inhibotyr - excitatory
    Sie(1+Ne-((numPools-1)/numPools)*Ne+Nre*(i-1):Ne-((numPools-1)/numPools)*Ne+Nre*i,1+Nci*(i-1):Nci*i) = ie;
end
See(1:Nre,(numPools-1)*Nce+1:numPools*Nce) = ee; % 0.125
Sei((numPools-1)*Nri+1:numPools*Nri,(numPools-1)*Nce+1:numPools*Nce) = 0.13;
%See((numPools-1)*Nre+1:numPools*Nre,(numPools-1)*Nce+1:numPools*Nce) = 0.02;
Sie(1:Nre,(numPools-1)*Nci+1:numPools*Nci) = ie;

%% Variable and parameter initialization. 
% Don't need to change anything here

simtime = 1000; % milliseconds

firingse=[];             % spike timings
firingsi=[];
firingsq=[];

dt = 1; % fraction of millisecond that the simulation steps each cycle

% voltage value variables for plotting
vq_view = zeros(Nq,simtime/dt);
ve_view = zeros(Ne,simtime/dt);
vi_view = zeros(Ni,simtime/dt);
uq_view = zeros(1,simtime/dt);

AMPA = 0.5;
GABAa = 0.5;

% build the postsynaptic voltage functions
%AMPA: fast, ionotropic, excitatory
%NMDA: slow, ionotropic, excitatory
%GABAa:fast, ionotropic, inhibitory
%GABAb:metabotropic, inhibitory
NMDAe = 0:(100/dt);
NMDAe = 0.5*exp(-NMDAe*dt*0.7)+0.5*exp(-NMDAe*dt*0.05);
AMPAe = 0:(100/dt);
AMPAe = exp(-AMPAe*dt*0.7);
GABAae = 0:(100/dt);
GABAae = 0.5*exp(-GABAae*dt*0.7)+0.5*exp(-GABAae*dt*0.05);
GABAbe = 0:(100/dt);
GABAbe = exp(-GABAbe*dt*0.7);

% values for postsynaptic potential changes
S_E = AMPA*AMPAe + (1-AMPA)*NMDAe;
S_I = GABAa*GABAae + (1-GABAa)*GABAbe;

figure; plot(S_E,'LineWidth',3); xlim([0 100]);
title('Postsynaptic Potential');
xlabel('Time(ms)');
ylabel('Synaptic Output');

%% Synaptic delays balance the overall network
% synaptic delays in ms
ex_syn_delay = Ne+100; %Q: Why use the number of neurons as param for delay? 
maxdelay = 35;
delaye = [5,30]; % [5,33];
delayi = [1,15]; % [1,15];
ax_delaye = randi(delaye,Ne); % Create NexNe matrix of integers sampled from uniform distribution 1:44
ax_delayi = randi(delayi,Ne,Ni); %Q: I'm a bit confused about why setting this randi matrix, and also what the below current/input means

% current/input variables
Ie = zeros(Ne,simtime/dt+size(NMDAe,2)+maxdelay/dt);
Ii = zeros(Ni,simtime/dt+size(NMDAe,2)+maxdelay/dt);
Iq = zeros(Nq,simtime/dt+size(NMDAe,2)+maxdelay/dt);

%% Main Body
% New inputs are determined by the sum of excitatory and inhibitory outputs
% of fired neurons scaled by the synaptic weights. 
%Q: what does new input mean, just pure stimulations?

for t=1:((1/dt)*simtime) % t steps every dt, and simtime is total ms simulated

  % define the external input into the first excitatory population
  if t < 80/dt && t > 40
%     % Input into separate populations with normal distribution
    Ie(1:Ne/numPools,t) = 1.5*ones(Ne/numPools,1);
  end
  
  % add noise into inputs
  Ie(:,t) = Ie(:,t)+1*randn(Ne,1); % background noise input
  Ii(:,t) = Ii(:,t)+1*randn(Ni,1);
  Iq(:,t) = Iq(:,t)+1*randn(Nq,1);
  %Ii=2*randn(Ni,1);  
  
  % Uncomment this to have an impulse input for 'active readout'
%   if t > 200 && t < 250
%       Iq(:,t) = Iq(:,t) + 1*ones(Nq,1); 
%   end
  
  % indices of spikes
  firede=find(ve>=V_th); % indices of spikes
  firedi=find(vi>=V_th);
  firedq=find(vq>=V_th);
  
  % keep a running list of spike times %Q: are these the neurons that keep
  % firing? Or what does this mean?
  firingse=[firingse; t+0*firede,firede];
  firingsi=[firingsi; t+0*firedi,firedi];
  firingsq=[firingsq; t+0*firedq,firedq];

  % initialize voltages
  ve(firede)=V_reset;
  vi(firedi)=V_reset;
  vq(firedq)=V_reset;
  
  % for every fired inhibitory neuron, add the IPSPs to the input variables 
  if (~isempty(firede))  
    for i = 1:size(firede) % number of fired neurons in IT
        % Calculate Input Currents
        n = firede(i,1);        
        for j = 1:Ne % cycle through synapses along rows of excitatory populations
            % 100 is used because that is the length of the synaptic
            % exponential matrix
            z = ax_delaye(j,n);
            Ie(j,t+z:t+z+size(S_E,2)-1) = Ie(j,t+z:t+z+size(S_E,2)-1)+S_E*See(j,n);
        end
        % inhibitory neurons
        for j = 1:Ni
            z = ax_delaye(j,n);
            Ii(j,t+z:t+z+size(S_I,2)-1) = Ii(j,t+z:t+z+size(S_I,2)-1)+S_I*Sei(j,n);
        end
        
        % downstream second population
        for j = 1:Nq
            Iq(j,t+z:t+z+size(S_E,2)-1) = Iq(j,t+z:t+z+size(S_E,2)-1)+S_E*Seq(j,n);
        end
    end
  end
  
  % for every fired inhibitory neuron, add the IPSPs to the input variables
  if ~isempty(firedi)
    for i = 1:size(firedi) % number of fired neurons in IN
            n = firedi(i,1);
        for j = 1:Ne % cycle through synapses along rows of inhibitory populations
            z = ax_delayi(j,n);
            Ie(j,t+z:t+z+size(S_I,2)-1) = Ie(j,t+z:t+z+size(S_I,2)-1)+S_I*Sie(j,n);    
        end
        for j = 1:Nq
            Iq(j,t+z:t+z+size(S_I,2)-1) = Iq(j,t+z:t+z+size(S_I,2)-1)+S_I*Siq(j,n);
        end
    end
  end
  
  V_infe = E_L + Ie(:,t)*R_m; %value that V_vect is exponentially
  V_infi = E_L + Ii(:,t)*R_m; %decaying towards at this time step    
  V_infq = E_L + Iq(:,t)*R_m;
  
  % this updates the voltages based on the inputs
  ve = V_infe + (ve - V_infe)*exp(-dt/tau); 
  vi = V_infi + (vi - V_infi)*exp(-dt/tau);
  vq = V_infq + (vq - V_infq)*exp(-dt/tau);
  
  ve_view(:,t) = ve;
  vi_view(:,t) = vi;
  vq_view(:,t) = vq;
end


%% Plot Output

figure;
subplot(4,1,1);
plot(firingse(:,1),firingse(:,2),'.'); ylabel('Neuron index'); title('Excitatory Neurons Ordered'); %Q: what does it mean by a neuron is ordered?
subplot(4,1,2);
plot(firingse(randperm(length(firingse)),1),firingse(:,2),'.'); title('Excitatory Neurons Shuffled');
subplot(4,1,3);
plot(firingsi(:,1),firingsi(:,2),'.r'); title('Inhibitory Neurons Ordered');
subplot(4,1,4);
plot(firingsi(randperm(length(firingsi)),1),firingsi(:,2),'.r'); title('Inhibitory Neurons Shuffled');
xlabel('Time (ms)');

figure;
% subplot(3,1,1);
% plot(Iq(1,1:simtime/dt)); title('Current(mA)');
subplot(2,1,1);
plot(vq_view'); title('Effect of Synaptic Strength'); xlabel('Time(ms)'); ylabel('Membrane Potential (mV)');
subplot(2,1,2);
plot(firingsq(:,1),firingsq(:,2),'.k'); title('Downstream Neurons'); xlim([0 1000]); ylim([0 Nq+1]); xlabel('Time(ms)'); ylabel('Neuron Index');
% avgdelay = [sum(sum(ax_delaye))/prod(size(ax_delaye)),sum(sum(ax_delayi))/prod(size(ax_delayi))]
